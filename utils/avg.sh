#! /usr/bin/env bash

if [ $# != 3 ]; then
    echo "usage: ${0} [best|latest] ckpt_dir avg_num"
    exit -1
fi

avg_mode=${1} # best,latest
ckpt_dir=${2}
average_num=${3}
decode_checkpoint=${ckpt_dir}/avg_${average_num}.pdparams

if [ $avg_mode == best ];then
    # best
    avg_model.py \
    --dst_model ${decode_checkpoint} \
    --ckpt_dir ${ckpt_dir}  \
    --num ${average_num} \
    --val_best
else
    # latest
    avg_model.py \
    --dst_model ${decode_checkpoint} \
    --ckpt_dir ${ckpt_dir}  \
    --num ${average_num}
fi

if [ $? -ne 0 ]; then
    echo "Failed in avg ckpt!"
    exit 1
fi

exit 0
