#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../lite_predict.py \
        --inference_dir=${train_output_path}/pdlite \
        --am=fastspeech2_aishell3 \
        --voc=pwgan_aishell3 \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/lite_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=0
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../lite_predict.py \
        --inference_dir=${train_output_path}/pdlite \
        --am=fastspeech2_aishell3 \
        --voc=hifigan_aishell3 \
        --text=${BIN_DIR}/../sentences.txt \
        --output_dir=${train_output_path}/lite_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=0
fi
