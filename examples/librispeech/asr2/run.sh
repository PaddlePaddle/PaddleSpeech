#!/bin/bash

set -e

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

gpus=0,1,2,3,4,5,6,7
stage=0
stop_stage=50
conf_path=conf/transformer.yaml
decode_conf_path=conf/decode/decode_base.yaml
dict_path=data/lang_char/train_960_unigram5000_units.txt
avg_num=10

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path}  ${ckpt}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh latest exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # attetion resocre decoder
    ./local/test.sh ${conf_path} ${decode_conf_path} ${dict_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # join ctc decoder, use transformerlm to score
    ./local/recog.sh  --ckpt_prefix exp/${ckpt}/checkpoints/${avg_ckpt}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # ctc alignment of test data
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} ${dict_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ./local/cacu_perplexity.sh || exit -1
fi

if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ]; then
    # export ckpt avg_n
    ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
fi


