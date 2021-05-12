#! /usr/bin/env bash

if [ $# != 2 ]; then
    echo "usage: ${0} ckpt_dir avg_num"
    exit -1
fi

ckpt_dir=${1}
average_num=${2}
decode_checkpoint=${ckpt_dir}/avg_${average_num}.pdparams

python3 -u ${MAIN_ROOT}/utils/avg_model.py \
--dst_model ${decode_checkpoint} \
--ckpt_dir ${ckpt_dir}  \
--num ${average_num} \
--val_best

if [ $? -ne 0 ]; then
    echo "Failed in avg ckpt!"
    exit 1
fi

exit 0
