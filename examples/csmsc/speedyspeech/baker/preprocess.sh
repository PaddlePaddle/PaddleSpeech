#!/bin/bash

stage=0
stop_stage=100

export MAIN_ROOT=`realpath ${PWD}/../../../`

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # get durations from MFA's result
    echo "Generate durations.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=./baker_alignment_tone \
        --output=durations.txt \
        --config=conf/default.yaml
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Extract features ..."
    python3 ../preprocess.py \
        --dataset=baker \
        --rootdir=~/datasets/BZNSYP/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=conf/default.yaml \
        --num-cpu=20 \
        --cut-sil=True \
        --use-relative-path=True
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Get features' stats ..."
    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="feats" \
        --use-relative-path=True
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize and covert phone/tone to id, dev and test should use train's stats
    echo "Normalize ..."
    python3 ../normalize.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --dumpdir=dump/train/norm \
        --stats=dump/train/feats_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --tones-dict=dump/tone_id_map.txt \
        --use-relative-path=True

    python3 ../normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --stats=dump/train/feats_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --tones-dict=dump/tone_id_map.txt \
        --use-relative-path=True

    python3 ../normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --stats=dump/train/feats_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --tones-dict=dump/tone_id_map.txt \
        --use-relative-path=True

fi
