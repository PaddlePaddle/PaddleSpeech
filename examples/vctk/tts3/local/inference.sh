#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_vctk \
        --voc=pwgan_vctk \
        --text=${BIN_DIR}/../sentences_en.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=0 \
        --lang=en
fi

