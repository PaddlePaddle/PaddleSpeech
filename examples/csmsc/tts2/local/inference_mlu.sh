#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

# for more GAN Vocoders
# multi band melgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=speedyspeech_csmsc \
        --voc=mb_melgan_csmsc \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --device mlu
fi

# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=speedyspeech_csmsc \
        --voc=hifigan_csmsc \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --tones_dict=dump/tone_id_map.txt \
        --device mlu
fi
