from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
from datetime import datetime
import math
import numpy as np
import re
import logging
import random

from pytorch_pretrained_bert import BertTokenizer, BertModel

from mylib.modified_bert_text_data import BertTextPairIterator, BertTextIterator
#from mylib.text_data import TextPairIterator, TextIterator
from mylib.utils import timeSince, ids2words, unbpe
from mylib.layers import CudaVariable
from mylib.opts import ScheduledOptim

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#import nmt_const as Const
import bert_const as Const
from scratch_model import BertAttNMT
from scratch_trans import translate_file
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from visdom import Visdom

class VisdomLinePlotter(object):
    def __init__(self, env_name='main', port=8095):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
        self.updatetextwindow = self.viz.text('Label and Prediction \n', env=self.env)
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iters',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
    def textboard(self, strs):
        self.viz.text(strs, win=self.updatetextwindow, append=True, env=self.env)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def train_decoder(model, optimizer, x_data, x_mask, x_segm, y_data, y_mask, args, logger=None):
    print('train decoder first')
    return loss
def train_bert(model, optimizer, x_data, x_mask, x_segm, y_data, y_mask, args, logger=None):
    print('train bert')
    return loss
def train(model, optimizer, x_data, x_mask, x_segm, y_data, y_mask, args, logger=None):
    model.train()
    #model.eval()
    loss = model(x_data, x_mask, x_segm, y_data, y_mask, logger=logger)
    optimizer.zero_grad()
    loss.backward()
    #if args.grad_clip > 0:
    #    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss
    #return 0
def train_model(args):
    now  = datetime.now()
    global plotter
    plotter = VisdomLinePlotter(env_name='BERT+NMT', port=8095)
    t = now.strftime('%Y_%m%d_%H%M:%S')
    log_filename = 'bert_batch_'+str(args.batch_size) + '_' + t + '.log'
    logger = logging.getLogger('DEC')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    stream_handler=logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_filename)
    logger.addHandler(file_handler)
    logger.info("LOGGING START")

    start = time.time()
    if args.using_pretrained:
        logger.info('load trained Decoder')
        model = torch.load(args.save_dir + '/trained_decoder768.pth')
        model.cuda()
    else:
        logger.info('training from scratch')
        model = BertAttNMT(device, args).cuda()
    train_iter = BertTextPairIterator(args.train_src_file, args.train_trg_file, tokenizer,
                        batch_size=args.batch_size, maxlen=args.max_length,
                        ahead=1000, const_id=Const)
    # Predict hidden states features for each layer
    logger.info('start training')
    loss_total =  0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    if args.detach_bert:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if str(name).startswith('bert'):
                    param.requires_grad=False
                else:
                    print(str(name))
        logger.info('disable finished')
    for x_data, x_mask, x_segm, y_data, y_mask, cur_line, iloop in train_iter:
        loss = train(model, optimizer, x_data, x_mask, x_segm, y_data, y_mask, args, logger)
        loss_total += loss
        if iloop == args.max_iter and args.detach_bert:
             for name, param in model.named_parameters():
                 if str(name).startswith('bert'):
                     param.requires_grad=True
             logger.info('enable encoder after' + str(args.max_iter) + 'iters')
             params = filter(lambda p: p.requires_grad, model.parameters())
             optimizer = optim.Adam(params, lr=0.000005)
             logger.info('optimizer setting with learing rate 0.000005')
             filename = args.save_dir + '/trained_decoder' + str(args.dim_dec) + '.pth'
             torch.save(model, filename)
        if iloop % args.print_every == 0:
            loss_avg = loss_total/args.print_every
            loss_total = 0
            if args.visdom:
                plotter.plot('training', 'train_loss', str(args.dim_dec) + 'dim and batch : ' + str(args.batch_size), iloop, loss_avg.item())
            if iloop > args.max_iter and args.using_pretrained:
                logger.info('[DECODER] %d iteration, Loss : %.4f, Time : %s, Ppl: %8.2f' % (iloop, loss_avg, timeSince(start), math.exp(loss_avg)))
            else:
                logger.info('[BERT+DECODER] %d iteration, Loss : %.4f, Time : %s, Ppl: %8.2f' % (iloop, loss_avg, timeSince(start), math.exp(loss_avg)))

        if iloop % args.valid_every == 0 and iloop >= args.valid_start:
            file_name = args.save_dir + '/' + args.model_file + '.pth'
            logger.info('saving the model to '+file_name + 'with loss : ' + str(loss_avg.item()))
            torch.save(model, file_name)
            eval_model(args, model, iloop, file_name, logger)

def eval_model(args, model, iloop, file_name, logger):
    model.eval()
    logger.info('start eval')
    with torch.no_grad():
        bleu_score = translate_file(tokenizer, model, args, valid=True, logger=logger)
    if os.path.exists(file_name + '.bleu'):
        with open(file_name+'.bleu', 'r') as bfp:
            lines = bfp.readlines()
            prev_bleus = [float(bs.split()[1]) for bs in lines]
    else:
        prev_bleus=[0]
    if bleu_score >= np.max(prev_bleus):
        torch.save(model, file_name + '.best.pth')
    mode = 'w' if iloop == args.valid_start else 'a'
    with open(file_name + '.bleu', mode) as bfp:
        bfp.write(str(iloop) + '\t' + str(bleu_score) + '\n')
    logger.info('---- validation bleu_score : ' + str(bleu_score))
