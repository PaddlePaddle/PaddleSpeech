#!/bin/bash
source path.sh
set -e

gpus=4,5,6,7
stage=3
stop_stage=3
conf_path=conf/rnnt.yaml
ips=            #xx.xx.xx.xx,xx.xx.xx.xx
decode_conf_path=conf/tuning/decode.yaml
avg_num=10
audio_file=data/demo_01_03.wav

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
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt} ${ips}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=2 ./local/test.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # ctc alignment of test data
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi

# Optionally, you can add LM and test it with runtime.
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # test a single .wav file
    CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} ${decode_conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
fi

# Not supported at now!!!
if [ ${stage} -le 51 ] && [ ${stop_stage} -ge 51 ]; then
     # export ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
fi

# Need further installation! Read the install.md to complete further installation
if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
    echo "warning: deps on kaldi and srilm, please make sure installed."
    # train lm and build TLG
    ./local/tlg.sh --corpus aishell --lmtype srilm
fi
