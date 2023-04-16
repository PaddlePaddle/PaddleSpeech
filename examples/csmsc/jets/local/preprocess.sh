#!/bin/bash
set -e
stage=0
stop_stage=100

config_path=$1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # get durations from MFA's result
    echo "Generate durations.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=./baker_alignment_tone \
        --output=durations.txt \
        --config=${config_path}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # extract features
    echo "Extract features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=baker \
        --rootdir=~/datasets/BZNSYP/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True \
        --token_average=True
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="feats"

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
        --feats-stats=dump/train/feats_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --feats-stats=dump/train/feats_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt

    python3 ${BIN_DIR}/normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --feats-stats=dump/train/feats_stats.npy \
        --pitch-stats=dump/train/pitch_stats.npy \
        --energy-stats=dump/train/energy_stats.npy \
        --phones-dict=dump/phone_id_map.txt \
        --speaker-dict=dump/speaker_id_map.txt
fi
