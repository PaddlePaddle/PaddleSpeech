#!/bin/bash

stage=0
stop_stage=100

config_path=$1
datasets_root_dir=$2
mfa_root_dir=$3

# 1. get durations from MFA's result
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Generate durations_baker.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=${mfa_root_dir}/baker_alignment_tone \
        --output durations_baker.txt \
        --config=${config_path}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Generate durations_ljspeech.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=${mfa_root_dir}/ljspeech_alignment \
        --output durations_ljspeech.txt \
        --config=${config_path}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Generate durations_aishell3.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=${mfa_root_dir}/aishell3_alignment_tone \
        --output durations_aishell3.txt \
        --config=${config_path}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Generate durations_vctk.txt from MFA results ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
        --inputdir=${mfa_root_dir}/vctk_alignment \
        --output durations_vctk.txt \
        --config=${config_path}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # concat duration file
    echo "concat durations_baker.txt, durations_ljspeech.txt, durations_aishell3.txt and durations_vctk.txt to durations.txt"
    cat durations_baker.txt durations_ljspeech.txt durations_aishell3.txt durations_vctk.txt > durations.txt
fi

# 2. extract features
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Extract baker features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=baker \
        --rootdir=${datasets_root_dir}/BZNSYP/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True \
        --write_metadata_method=a
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Extract ljspeech features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=ljspeech \
        --rootdir=${datasets_root_dir}/LJSpeech-1.1/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True \
        --write_metadata_method=a
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Extract aishell3 features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=aishell3 \
        --rootdir=${datasets_root_dir}/data_aishell3/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True \
        --write_metadata_method=a
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Extract vctk features ..."
    python3 ${BIN_DIR}/preprocess.py \
        --dataset=vctk \
        --rootdir=${datasets_root_dir}/VCTK-Corpus-0.92/ \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=${config_path} \
        --num-cpu=20 \
        --cut-sil=True \
        --write_metadata_method=a
fi


# 3. get features' stats(mean and std)
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
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


# 4. normalize and covert phone/speaker to id, dev and test should use train's stats
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
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
