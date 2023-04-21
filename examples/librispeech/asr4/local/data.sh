#!/bin/bash

stage=-1
stop_stage=100

unit_type=char
dict_dir=data/lang_char

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
mkdir -p ${dict_dir}
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

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

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # compute mean and stddev for normalizer
    num_workers=$(nproc)
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.train.raw" \
    --num_samples=2000 \
    --spectrum_type="fbank" \
    --feat_dim=161 \
    --delta_delta=false \
    --sample_rate=16000 \
    --stride_ms=10 \
    --window_ms=25 \
    --use_dB_normalization=False \
    --num_workers=${num_workers} \
    --output_path="data/mean_std.json"

    if [ $? -ne 0 ]; then
        echo "Compute mean and stddev failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # build vocabulary
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
    --unit_type ${unit_type} \
    --count_threshold=0 \
    --vocab_path="${dict_dir}/vocab.txt" \
    --manifest_paths="data/manifest.train.raw"

    if [ $? -ne 0 ]; then
        echo "Build vocabulary failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for set in train dev test dev-clean dev-other test-clean test-other; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
        --cmvn_path "data/mean_std.json" \
        --unit_type ${unit_type} \
        --vocab_path="${dict_dir}/vocab.txt" \
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    mkdir -p exp/hubert
    echo "Pretrained hubert model download"
    wget -P exp/hubert https://paddlespeech.bj.bcebos.com/hubert/hubert-large-lv60.pdparams
fi

exit 0