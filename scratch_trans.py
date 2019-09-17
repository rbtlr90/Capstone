# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import copy
import os
from io import open
import time
import re
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from mylib.text_data import TextIterator, TextPairIterator, read_dict
from mylib.utils import timeSince, ids2words, bert_unbpe
from mylib.layers import CudaVariable
from mylib.modified_bert_text_data import BertTextPairIterator, BertTextIterator
from pytorch_pretrained_bert import BertTokenizer, BertModel
#import nmt_const as Const
import bert_const as Const
from Beam import Beam

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def translate_nmt(model, x_data, x_segm, args):
    ctx, yt, ht, ct = model.reply_encode(x_data, x_segm)
    #ctx, yt, ht, ct = model(x_data, x_mask, y_data, y_mask, args)

    pos_idx = torch.from_numpy(np.arange(args.max_length+2)).type(torch.cuda.LongTensor)
    if args.beam_width == 1:
        y_hat = []
        for yi in range(args.max_length+2):
            prob, yt, ht, ct = model.dec_step(pos_idx[yi], ctx, yt, ht, ct)
            y_hat.append(yt)
            if yt[0] == Const.SEP:
                break
        y_hat = torch.stack(y_hat)
        y_hat = y_hat.cpu().numpy().flatten().tolist()

        return y_hat

    # for beam size > 1
    sample_sent = []
    sample_score = []

    k = args.beam_width
    live_k = 1 
    dead_k = 0 

    hyp_samples = [[]]
    hyp_scores = CudaVariable(torch.zeros(live_k,))
    hyp_states_h = []
    hyp_states_c = []
    
    for yi in range(args.max_length+2):
        ctx_k = ctx.expand(ctx.size(0), live_k, ctx.size(2))
        pt, yt, ht, ct = model.dec_step(pos_idx[yi], ctx_k, yt, ht, ct)

        cand_scores = hyp_scores.unsqueeze(1).expand(live_k, pt.size(1)) - pt
        cand_flat = cand_scores.view(-1)
        values, ranks_flat = torch.sort(cand_flat)
        ranks_flat = ranks_flat[:(k-dead_k)]

        voc_size = pt.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = Variable(torch.zeros(k-dead_k)).cuda()
        new_hyp_states_h = []
        new_hyp_states_c = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            ti = int(ti)
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = costs[idx]
            new_hyp_states_h.append(ht[ti])
            new_hyp_states_c.append(ct[ti])

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states_h = []
        hyp_states_c = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1].cpu().numpy() == Const.EOS: # EOS
                sample_sent.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states_h.append(new_hyp_states_h[idx])
                hyp_states_c.append(new_hyp_states_c[idx])
        
        live_k = new_live_k
        if new_live_k > 0:
            hyp_scores = torch.stack(hyp_scores)
        else:
            break
        if dead_k >= k:
            break

        yt = torch.stack([w[-1] for w in hyp_samples])
        ht = torch.stack(hyp_states_h)
        ct = torch.stack(hyp_states_c)

    # dump every remaining one
    if live_k > 0:
        for idx in range(live_k):
            sample_sent.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    # length normalization
    scores = [score/len(sample) for (score, sample) in zip(sample_score, sample_sent)]
    scores = torch.stack(scores).cpu().detach().numpy()
    best_sample = sample_sent[scores.argmin()]

    y_hat = torch.stack(best_sample)
    y_hat = y_hat.cpu().numpy().flatten().tolist()

    return y_hat
def cleanConst(str_text):
    str_text = str_text.replace('[CLS]', '').strip()
    str_text = str_text.replace('[SEP]', '').strip()
    str_text = str_text.replace('##', '')
    return str_text
def translate_file(tokenizer, model, args, valid=None, logger=None):
    logger.info('entering translating file')
    label_file = open(args.valid_trg_file, 'r')
#    question_file = open('/home/korchris/bert_scratch/data/question_valid.txt', 'r')
    question_file = open('/home/jks90/bert_scratch3/data/qvalid_wp_tok.txt', 'r')
    labels = label_file.readlines()
    questions = question_file.readlines()
    valid_iter = BertTextIterator(args.valid_src_file, tokenizer, maxlen=args.max_length, batch_size=1, ahead=1, const_id=Const)
    #trg_inv_dict = dict()
    #for kk, vv in trg_dict2.items():
    #    trg_inv_dict[vv] = kk

    if valid:
        multibleu_cmd = ["perl", args.bleu_script, args.valid_trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, 
                                universal_newlines=True)
    else:
        fp = open(args.trans_file, 'w')
    input_history = []
    history = []
    for x_data, x_mask, x_segm, cur_line, iloop in valid_iter:
        samples = translate_nmt(model, x_data, x_segm, args)
        #sentence = ids2words(trg_inv_dict, samples, eos_id=Const.EOS)
        #sentence = unbpe(sentence)
        sentence = tokenizer.convert_ids_to_tokens(samples)
        #print(sentence)
        sentence = bert_unbpe(sentence)
        history.append(sentence)
        if valid:
            mb_subprocess.stdin.write(sentence + '\n')
            mb_subprocess.stdin.flush()
        else:
            fp.write(sentence+'\n')
            if iloop % 500 == 0:
                print(iloop, 'is translated...')
    print('iloop done')
    for i in range(20):
        x = random.randint(0, len(questions)-1)
        print(i)
        logger.info(str(x))
        logger.info('[question]')
        logger.info(questions[x].rstrip('\n'))
        logger.info('[label]')
        logger.info(cleanConst(labels[x].rstrip('\n')))
        #logger.info('[input]')
        #logger.info(input_history[x])
        logger.info('[prediction]')
        logger.info(history[x])
    ret = -1
    if valid:
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        mb_subprocess.terminate()
        if out_parse:
            ret = float(out_parse.group()[6:])
    else:
        fp.close()
    torch.set_grad_enabled(True)
    logger.info('translate file done')
    return ret
