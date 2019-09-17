import argparse
import pickle as pkl

import torch

from scratch_main import train_model
#from scratch_main import eval_model
#from nmt_trans2 import translate_file 

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--logging", type=int, default=1)
parser.add_argument("--using_pretrained", type=int, default=1)
parser.add_argument("--detach_bert", type=int, default=1)
parser.add_argument("--model", type=str, default='nmt')
parser.add_argument("--model_file", type=str, default='')
parser.add_argument("--train_src_file", type=str, default='')
parser.add_argument("--train_trg_file", type=str, default='')
parser.add_argument("--valid_src_file", type=str, default='')
parser.add_argument("--valid_trg_file", type=str, default='')
parser.add_argument("--trans_file", type=str, default='')
parser.add_argument("--src_dict", type=str, default='')
parser.add_argument("--trg_dict", type=str, default='')
parser.add_argument("--visdom", default=False)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=100000)
parser.add_argument("--bleu_script", type=str, default='multi-bleu.perl')
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--grad_clip", type=float, default=0.0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout_p", type=float, default=0.0)
parser.add_argument("--dim_wemb", type=int, default=0)
parser.add_argument("--dim_att", type=int, default=0)
parser.add_argument("--dim_dec", type=int, default=0)
parser.add_argument('--bert_enc', type=int, default=768, help="random seed for initialization")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--valid_start", type=int, default=50)
parser.add_argument("--valid_every", type=int, default=50)
parser.add_argument("--train", type=int, default=0)
parser.add_argument("--trans", type=int, default=0)
parser.add_argument("--use_best", type=int, default=0)
parser.add_argument("--beam_width", type=int, default=1)

parser.add_argument("--output_dir", default="/home/jks90/bert_scratch3/BERT_RESULT", type=str,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument('--save-data', type=int, default=1, metavar='N',
                                    help='load data from pickle or not')
parser.add_argument('--load-data', type=int, default=1, metavar='N',
                                    help='load data from pickle or not')
parser.add_argument("--local-rank", default=-1, type=int, help="local rank for distributed training on gpus")
parser.add_argument("--no_cuda", default=False, action='store_true',help="Whether not to use CUDA when available")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

args = parser.parse_args()

# training
if args.train:
    print(args)
    print ('Training...')
    pkl.dump(args, open(args.save_dir+'/'+args.model_file+'.args.pkl', 'wb'), -1)
    with open(args.save_dir+'/'+args.model_file+'.args', 'w') as fp:
        for key in vars(args):
            fp.write(key + ': ' + str(getattr(args, key)) + '\n')

    args.beam_width = 1
    train_model(args)
if args.trans:
    print ('Translating...')
    file_name = args.save_dir + '/' + args.model_file + '.pth'
    if args.use_best == 1:
        file_name = file_name + '.best.pth' 
    model = torch.load(file_name)

    old_args = pkl.load(open(args.save_dir+'/'+args.model_file+'.args.pkl', 'rb'))
    args.model = old_args.model
    args.dim_dec = old_args.dim_dec
    
    ret = translate_file(model, args, valid=False)

print ('Done')
