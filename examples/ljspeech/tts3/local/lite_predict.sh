#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../lite_predict.py \
        --inference_dir=${train_output_path}/pdlite \
        --am=fastspeech2_ljspeech \
        --voc=pwgan_ljspeech \
        --text=${BIN_DIR}/../../assets/sentences_en.txt \
        --output_dir=${train_output_path}/lite_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --lang=en
fi

# hifigan
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../lite_predict.py \
        --inference_dir=${train_output_path}/pdlite \
        --am=fastspeech2_ljspeech \
        --voc=hifigan_ljspeech \
        --text=${BIN_DIR}/../../assets/sentences_en.txt \
        --output_dir=${train_output_path}/lite_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --lang=en
fi
