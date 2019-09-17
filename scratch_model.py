import numpy as np
import math
import random
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from mylib.layers import CudaVariable, myEmbedding, myLinear, myLSTM, biLSTM
from pytorch_pretrained_bert.modeling import BertModel

class BertAttNMT(nn.Module):
    def __init__(self, device, args=None):
        super(BertAttNMT, self).__init__()
        self.device = device
        self.bert_enc = 768
        self.dim_dec = args.dim_dec
        self.dim_wemb = 768
        self.dim_att = 600
        self.trg_words_n = 31199

        self.dec_emb = myEmbedding(self.trg_words_n, self.dim_wemb)

        self.bert=BertModel.from_pretrained('bert-base-uncased')
        self.bert.to(self.device)
        #self.dec_emb = self.bert.BertEmbeddings(self.trg_words_n, self.dim_wemb)
        self.dec_h0 = myLinear(self.bert_enc, self.dim_dec)
        self.dec_c0 = myLinear(self.bert_enc, self.dim_dec)


        self.att1 = myLinear(self.bert_enc + self.dim_wemb + self.dim_dec, self.dim_att)
        self.att2 = myLinear(self.dim_att, 1)

        self.rnn_step = nn.LSTMCell(self.dim_wemb + self.bert_enc, self.dim_dec)
        self.readout = myLinear(self.bert_enc + self.dim_dec + self.dim_wemb, self.dim_wemb*2)
        self.logitout = myLinear(self.dim_wemb, self.trg_words_n)
        self.logitout.weight = self.dec_emb.weight # weight tying.

    def bert_encoder(self, x_data, x_segm):
        ctx, _ = self.bert(x_data, x_segm, output_all_encoded_layers=False)
        return ctx

    def dec_step(self, yi, ctx, y_tm1, htm, ctm, xm=None):
        Tx, Bn, Ec = ctx.size()
        y_tm1 = y_tm1.view(Bn,) # Bn 1
        y_emb = self.dec_emb(y_tm1) # Bn E
        att_in = torch.cat((ctx, y_emb.expand(Tx, Bn, y_emb.size(1)) , htm.expand(Tx, Bn, htm.size(1))), dim=2)
        att1 = F.tanh(self.att1(att_in)) # Tx Bn E 
        att2 = self.att2(att1).view(Tx, Bn) # Tx Bn 
        att2 = att2 if xm is None else att2*xm # added new. should check!
        att2 = att2 - torch.max(att2) 

        alpha = torch.exp(att2) if xm is None else torch.exp(att2)*xm
        alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
        ctx_t = torch.sum(alpha.unsqueeze(2) * ctx, dim=0) # Bn E 

        dec_in = torch.cat((y_emb, ctx_t), dim=1)
        ht, ct = self.rnn_step(dec_in, (htm, ctm))

        readin = torch.cat((ctx_t, ht, y_emb), dim=1)
        readout = self.readout(readin)
        readout = readout.view(readout.size(0), self.dim_wemb, 2) # Bn Wemb 2
        read_max = torch.max(readout, dim=2)[0] # Bn Wemb

        logit = self.logitout(read_max)
        prob = F.log_softmax(logit, dim=1)
        topv, yt = prob.topk(1)
        return prob, yt.view(Bn,), ht, ct

    def forward(self, x_data, x_mask, x_segm, y_data, y_mask, logger):
        x_data = CudaVariable(torch.LongTensor(x_data))
        x_mask = CudaVariable(torch.FloatTensor(x_mask))
        y_data = CudaVariable(torch.LongTensor(y_data))
        y_mask = CudaVariable(torch.FloatTensor(y_mask))
        x_segm = CudaVariable(torch.LongTensor(x_segm))
        T, B = x_data.size()
        x_data = x_data.view(B, T)
        x_segm = x_segm.view(B, T)
        ctx = self.bert_encoder(x_data, x_segm)
        Bn, Tx, Ec = ctx.size()
        ctx = ctx.view(Tx, Bn, Ec)
        #if self.training:
        #    logger.info('training')
        #else:
        #    logger.info('eval')
        #logger.info(ctx[:4, :, :10])
        #print('training')
        #print(ctx[:4, :, :10])
        loss=0
        Ty, Bn = y_data.size()
        ctx_sum = torch.sum(ctx*x_mask.unsqueeze(2), dim=0)
        ctx_mean = ctx_sum/torch.sum(x_mask, dim=0).unsqueeze(1)
        ht = F.tanh(self.dec_h0(ctx_mean))
        ct = F.tanh(self.dec_c0(ctx_mean))
        yt = CudaVariable(torch.zeros(Bn, )).type(torch.cuda.LongTensor)

        loss = 0
        criterion = nn.NLLLoss(reduce=False)
        pos_idx = torch.from_numpy(np.arange(Ty)).type(torch.cuda.LongTensor)
        for yi in range(Ty):
            prob, yt, ht, ct = self.dec_step(pos_idx[yi], ctx, yt, ht, ct, x_mask)
            loss_t = criterion(prob, y_data[yi])
            loss += torch.sum(loss_t * y_mask[yi])/Bn
            yt = y_data[yi]
        return loss

    def reply_encode(self, x_data, x_segm):
        x_data = CudaVariable(torch.LongTensor(x_data))
        x_segm = CudaVariable(torch.LongTensor(x_segm))
        T, B = x_data.size()
        x_data = x_data.view(B, T)
        x_segm = x_segm.view(B, T)
        ctx = self.bert_encoder(x_data, x_segm)
        Bn, Tx, Ec = ctx.size()
        ctx = ctx.view(Tx, Bn, Ec)
        #print('eval')
        #print(ctx[:4, :, :10])
        ctx_mean = torch.mean(ctx, dim=0)

        ht = F.tanh(self.dec_h0(ctx_mean)) # h0 
        ct = F.tanh(self.dec_c0(ctx_mean)) # c0 
        yt = CudaVariable(torch.zeros(1, )).type(torch.cuda.LongTensor) # y0, BOS=0

        return ctx, yt, ht, ct
