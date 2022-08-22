#!/bin/bash
set -e

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

gpus=0
stage=0
stop_stage=50
conf_path=conf/wav2vec2ASR.yaml
ips=            #xx.xx.xx.xx,xx.xx.xx.xx
decode_conf_path=conf/tuning/decode.yaml
avg_num=30
audio_file=data/demo_002_en.wav

. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi

