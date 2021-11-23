#!/bin/bash
set -e
source path.sh

gpus=0,1,2,3
stage=1
stop_stage=4
conf_path=conf/transformer_mtl_noam.yaml
ckpt_path=paddle.98
avg_num=5
data_path=./TED_EnZh # path to unzipped data
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    bash ./local/data.sh --data_dir ${data_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download pretrained
    bash ./local/download_pretrain.sh || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train_finetune.sh ${conf_path}  ${ckpt} ${ckpt_path}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi