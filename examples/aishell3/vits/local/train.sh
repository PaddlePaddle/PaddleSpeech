#!/bin/bash

config_path=$1
train_output_path=$2

# install monotonic_align
cd ${MAIN_ROOT}/paddlespeech/t2s/models/vits/monotonic_align
python3 setup.py build_ext --inplace
cd -

python3 ${BIN_DIR}/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=4 \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt
