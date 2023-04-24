#!/bin/bash

stage=0
stop_stage=100

config_path=$1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # extract features
    echo "Extract features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=opencpop \
        --rootdir=~/datasets/Opencpop/segments \
        --dumpdir=dump \
        --label-file=~/datasets/Opencpop/segments/transcriptions.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="speech"

    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="pitch"

    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="energy"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize and covert phone/speaker to id, dev and test should use train's stats
    echo "Normalize ..."
    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --dumpdir=dump/train/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # Get feature(mel) extremum for diffusion stretch
    echo "Get feature(mel) extremum  ..."
    python3 ${BIN_DIR}/get_minmax.py \
        --metadata=dump/train/norm/metadata.jsonl \
        --speech-stretchs=dump/train/speech_stretchs.npy
fi
