#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../inference_streaming.py \
        --inference_dir=${train_output_path}/inference_streaming \
        --am=fastspeech2_csmsc \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out_streaming \
        --phones_dict=dump/phone_id_map.txt \
        --am_streaming=True
fi

# for more GAN Vocoders
# multi band melgan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../inference_streaming.py \
        --inference_dir=${train_output_path}/inference_streaming \
        --am=fastspeech2_csmsc \
        --am_stat=dump/train/speech_stats.npy \
        --voc=mb_melgan_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out_streaming \
        --phones_dict=dump/phone_id_map.txt \
        --am_streaming=True
fi

# hifigan
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 ${BIN_DIR}/../inference_streaming.py \
        --inference_dir=${train_output_path}/inference_streaming \
        --am=fastspeech2_csmsc \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_csmsc \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/pd_infer_out_streaming \
        --phones_dict=dump/phone_id_map.txt \
        --am_streaming=True
fi

