#!/bin/bash

if [ $# != 1 ];then
    echo "usage: ${0} ckpt_dir"
    exit -1
fi

ckpt_dir=$1

stage=-1
stop_stage=100
unit_type=char

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

bash local/download_model.sh ${ckpt_dir}
if [ $? -ne 0 ]; then
   exit 1
fi

cd ${ckpt_dir}
tar xzvf librispeech_v1.8_to_v2.x.tar.gz
cd -
mv ${ckpt_dir}/mean_std.npz data/
mv ${ckpt_dir}/vocab.txt data/

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/librispeech/librispeech.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/librispeech" \
    --full_download="True"

    if [ $? -ne 0 ]; then
        echo "Prepare LibriSpeech failed. Terminated."
        exit 1
    fi

    for set in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
        mv data/manifest.${set} data/manifest.${set}.raw
    done

    rm -rf data/manifest.train.raw data/manifest.dev.raw  data/manifest.test.raw
    for set in train-clean-100 train-clean-360 train-other-500; do
        cat data/manifest.${set}.raw >> data/manifest.train.raw
    done

    for set in dev-clean dev-other; do
        cat data/manifest.${set}.raw >> data/manifest.dev.raw
    done

    for set in test-clean test-other; do
        cat data/manifest.${set}.raw >> data/manifest.test.raw
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for set in train dev test dev-clean dev-other test-clean test-other; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
        --cmvn_path "data/mean_std.npz" \
        --unit_type ${unit_type} \
        --vocab_path="data/vocab.txt" \
        --manifest_path="data/manifest.${set}.raw" \
        --output_path="data/manifest.${set}"

        if [ $? -ne 0 ]; then
            echo "Formt mnaifest.${set} failed. Terminated."
            exit 1
        fi
    }&
    done
    wait
fi

echo "LibriSpeech Data preparation done."
exit 0

