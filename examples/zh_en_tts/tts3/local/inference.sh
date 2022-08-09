#!/bin/bash

train_output_path=$1

stage=0
stop_stage=0

# voc: pwgan_aishell3
# the spk_id=174 means baker speaker, default
# the spk_id=175 means ljspeech speaker
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_mix \
        --voc=pwgan_aishell3 \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --lang=mix \
        --spk_id=174 
fi


# voc: hifigan_aishell3
# the spk_id=174 means baker speaker, default
# the spk_id=175 means ljspeech speaker
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 ${BIN_DIR}/../inference.py \
        --inference_dir=${train_output_path}/inference \
        --am=fastspeech2_mix \
        --voc=hifigan_aishell3 \
        --text=${BIN_DIR}/../sentences_mix.txt \
        --output_dir=${train_output_path}/pd_infer_out \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --lang=mix \
        --spk_id=174
fi
