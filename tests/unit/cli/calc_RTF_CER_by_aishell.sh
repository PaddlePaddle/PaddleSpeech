#!/bin/bash

source path.sh
stage=-1
stop_stage=100
model_name=conformer_online_aishell
gpus=5
log_file=res.log
res_file=res.rsl
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
    export CUDA_VISIBLE_DEVICES=${gpus}
    cat data/manifest.test | paddlespeech asr --model ${model_name} --device gpu --decode_method attention_rescoring --rtf -v &> ${log_file}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    cat ${log_file} | grep "^[0-9]" > ${res_file}
    python utils/compute-wer.py --char=1 --v=1 \
        data/manifest.test.text ${res_file} > ${res_file}.error
fi
