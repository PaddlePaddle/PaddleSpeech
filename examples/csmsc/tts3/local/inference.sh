#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_csmsc \
        --voc=pwgan_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt
fi

# for more GAN Vocoders
# multi band melgan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_csmsc \
        --voc=mb_melgan_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt
fi

# hifigan
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_csmsc \
        --voc=hifigan_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt
fi

# wavernn
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_csmsc \
        --voc=wavernn_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt
fi