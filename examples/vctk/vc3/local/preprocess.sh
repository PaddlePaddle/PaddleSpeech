#!/bin/bash

stage=0
stop_stage=100

config_path=$1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # extract features
    echo "Extract features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=vctk \
        --rootdir=~/datasets/VCTK-Corpus-0.92/ \
        --dumpdir=dump \
        --config=${config_path} \
        --num-cpu=20

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
echo "Normalize ..."
    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --dumpdir=dump/train/norm \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --speaker-dict=dump/speaker_id_map.txt

fi
