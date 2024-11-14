
#!/bin/bash

config_path=$1
train_output_path=$2
# export MLU_VISIBLE_DEVICES=8
python ${BIN_DIR}/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=0 \
    --nmlu=2 \
    --phones-dict=dump/phone_id_map.txt \
    --tones-dict=dump/tone_id_map.txt \
    --use-relative-path=True
