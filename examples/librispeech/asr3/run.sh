#!/bin/bash
set -e

MODEL=hubert
. ./path.sh ${MODEL} || exit 1;
. ./cmd.sh || exit 1;

gpus=2
stage=1
stop_stage=1
conf_path=conf/${MODEL}ASR.yaml
ips=            #xx.xx.xx.xx,xx.xx.xx.xx
decode_conf_path=conf/tuning/decode.yaml
avg_num=1
resume=         # xx e.g. 30

. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

audio_file=data/demo_002_en.wav

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
ckpt=test3
echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${resume} ${ips} 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # greedy search decoder
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # test a single .wav file
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
fi
