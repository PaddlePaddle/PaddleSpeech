#!/bin/bash

stage=1
stop_stage=100

export MAIN_ROOT=`realpath ${PWD}/../../../`

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # extract features
    echo "Extract features ..."
    python3 ../preprocess.py  \
        --dataset=ljspeech \
        --rootdir=~/datasets/LJSpeech-1.1/ \
        --dumpdir=dump \
        --config-path=conf/default.yaml \
        --num-cpu=8
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="speech"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize and covert phone to id, dev and test should use train's stats
    echo "Normalize ..."
    python3 ../normalize.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --dumpdir=dump/train/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ../normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ../normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --speech-stats=dump/train/speech_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt
fi
