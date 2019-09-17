import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from mylib.bert_text_data import BertTextPairIterator, BertTextIterator
from mylib.text_data import TextPairIterator, TextIterator
from mylib.layers import CudaVariable, myEmbedding, myLinear, myLSTM, biLSTM
import random
#import bert_const as Const
import nmt_const as Const
import re
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model.eval()
#model.to('cuda')
data_dir = '/home/korchris/bert_nmt_pytorch/bert_data/'
source = data_dir + 'd_sample.txt'
target = data_dir + 'a_sample.txt'
svoca = data_dir + 'dialogue_voca.pkl'
tvoca = data_dir + 'answer_voca.pkl'
voca_len1 = tokenizer.len_voca()
#train_iter = BertTextPairIterator(source, target, tokenizer, batch_size=1, max_len=250, ahead=100, const_id=Const)
#train_iter = BertTextIterator(source, tokenizer, batch_size=1, maxlen=300, ahead=1, const_id=Const)
text = "[CLS] michinnom daemeori mumamq Epxleiq  yo [SEP] chanung see [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed)
recover = tokenizer.convert_ids_to_tokens(indexed)
print(recover)
voca_len = tokenizer.len_voca()
print('before : ', voca_len1, ' after : ', voca_len)
#train_iter = TextPairIterator(source, target, svoca, tvoca, 
#        batch_size=1, maxlen=250, ahead=1000, resume_num=0, mask_pos=False, const_id=Const)
# Predict hidden states features for each layer
'''
with torch.no_grad():
    model.eval()
    for x_data, x_mask, cur_line, iloop in train_iter:
        x_data = CudaVariable(torch.LongTensor(x_data))
        T, B = x_data.size()
        x_data=x_data.view(B, T)
        encoded_layers, _ = model(x_data, output_all_encoded_layers=False)
        print(x_data)
        Bn, Tx, Ec = encoded_layers.size()
        ctx = encoded_layers.view(Tx, Bn, Ec)
        print(ctx[20:, :, :20])
'''
print('haha')
