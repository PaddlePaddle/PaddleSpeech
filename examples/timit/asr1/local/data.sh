#!/bin/bash

stage=-1
stop_stage=100

dict_dir=data/lang_char

unit_type=word
TIMIT_path=

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

mkdir -p data
mkdir -p ${dict_dir}
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/timit/timit_kaldi_standard_split.py \
    --manifest_prefix="data/manifest" \
    --src="data/local" \

    if [ $? -ne 0 ]; then
        echo "Prepare TIMIT failed. Terminated."
        exit 1
    fi
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # compute mean and stddev for normalizer
    num_workers=$(nproc)
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.train.raw" \
    --num_samples=-1 \
    --spectrum_type="fbank" \
    --feat_dim=80 \
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
    for set in train dev test; do
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

echo "TIMIT Data preparation done."
exit 0
