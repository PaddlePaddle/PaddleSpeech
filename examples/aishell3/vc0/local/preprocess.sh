#!/bin/bash

stage=0
stop_stage=100

input=$1
preprocess_path=$2
alignment=$3
ge2e_ckpt_path=$4

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../ge2e/inference.py \
        --input=${input} \
        --output=${preprocess_path}/embed \
        --device="gpu" \
        --checkpoint_path=${ge2e_ckpt_path}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Process wav ..."
    python3 ${BIN_DIR}/process_wav.py \
        --input=${input}/wav \
        --output=${preprocess_path}/normalized_wav \
        --alignment=${alignment}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 ${BIN_DIR}/preprocess_transcription.py \
        --input=${input} \
        --output=${preprocess_path}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 ${BIN_DIR}/extract_mel.py \
        --input=${preprocess_path}/normalized_wav \
        --output=${preprocess_path}/mel
fi
