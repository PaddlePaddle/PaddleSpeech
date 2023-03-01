#!/bin/bash

train_output_path=$1
add_blank=$2

stage=0
stop_stage=0

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=vits_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --add-blank=${add_blank}
fi