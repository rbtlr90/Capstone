USE_PRE=0
DETACH=1
MAX_ITER=100000
BATCH=16
LOGGING=1
DIM_WEMB=768
DIM_ATT=600 # better than 1000
DIM_DEC=400
BERT_ENC=768
DICT1='qtrain_wp_tok_voca.pkl'
DICT2='atrain_wp_tok_voca.pkl'
DATA_DIR=$HOME'/bert_scratch3/data2'
VALID_FILE1='qvalid_wp_tok.txt'
VALID_FILE2='avalid_wp_tok.txt'
#VALID_FILE1='d_sample.txt'
#VALID_FILE2='a_sample.txt'
TRAIN_FILE1='qtrain_wp_tok.txt'
TRAIN_FILE2='atrain_wp_tok.txt'
#Train 1 -> Bert data dialogue train set
TRAIN_FILE1=$DATA_DIR'/'$TRAIN_FILE1
#Train 2 -> answer data (without sep and cls)
TRAIN_FILE2=$DATA_DIR'/'$TRAIN_FILE2
VALID_FILE1=$DATA_DIR'/'$VALID_FILE1
VALID_FILE2=$DATA_DIR'/'$VALID_FILE2

DICT1=$DATA_DIR2'/'$DICT1
DICT2=$DATA_DIR'/'$DICT2

SAVE_DIR='./results'
#mkdir $SAVE_DIR
MODEL='BERT'
MODEL_FILE=$MODEL'.PRE=$USE_PRE'$DIM_WEMB'.'$DIM_ENC'.'$DIM_ATT'.'$DIM_DEC'.gpu'$1

CUDA_VISIBLE_DEVICES=$1 python3 scratch_run.py  --using_pretrained=$USE_PRE --train=1 \
        --model=$MODEL --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --logging=$LOGGING --batch_size=$BATCH --bert_enc=$BERT_ENC \
        --train_src_file=$TRAIN_FILE1 --train_trg_file=$TRAIN_FILE2 \
        --valid_src_file=$VALID_FILE1 --valid_trg_file=$VALID_FILE2 \
        --dim_wemb=$DIM_WEMB  --dim_att=$DIM_ATT --dim_dec=$DIM_DEC \
        --src_dict=$DICT1 --trg_dict=$DICT2 --detach_bert=$DETACH \
        --max_iter=$MAX_ITER --valid_start=10000 --print_every=500 --valid_every=10000
