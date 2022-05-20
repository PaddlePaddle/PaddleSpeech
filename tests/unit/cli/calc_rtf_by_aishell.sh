#!/bin/bash

source path.sh
stage=-1
stop_stage=100
MAIN_ROOT=../../..

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}
mkdir -p data

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 aishell_test_prepare.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/aishell"

    if [ $? -ne 0 ]; then
        echo "Prepare Aishell failed. Terminated."
        exit 1
    fi

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   cat data/manifest.test | paddlespeech asr --model conformer_online_aishell --device gpu --decode_method ctc_prefix_beam_search --rtf -v
fi
