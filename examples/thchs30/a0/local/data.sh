#! /usr/bin/env bash

stage=-1
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/thchs30/thchs30.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/thchs30"

    if [ $? -ne 0 ]; then
        echo "Prepare THCHS-30 failed. Terminated."
        exit 1
    fi
    
fi

echo "THCHS-30  data preparation done."
exit 0
