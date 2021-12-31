#!/bin/bash
set -e
source path.sh

stage=0
stop_stage=100
conf_path=conf/deepspeech2.yaml
decode_conf_path=conf/tuning/decode.yaml
avg_num=1
model_type=offline
gpus=1

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

v18_ckpt=librispeech_v1.8
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    mkdir -p exp/${ckpt}/checkpoints
    bash ./local/data.sh exp/${ckpt}/checkpoints || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=${gpus} ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${v18_ckpt} ${model_type}|| exit -1
fi
