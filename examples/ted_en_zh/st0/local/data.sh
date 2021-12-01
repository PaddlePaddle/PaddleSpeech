#!/bin/bash

set -e

stage=-1
stop_stage=100
dict_dir=data/lang_char

# bpemode (unigram or bpe)
nbpe=8000
bpemode=unigram
bpeprefix="${dict_dir}/bpe_${bpemode}_${nbpe}"
data_dir=./TED-En-Zh


. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;


TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}
mkdir -p data
mkdir -p ${dict_dir}


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e ${data_dir} ]; then
        echo "Error: ${data_dir} Dataset is not avaiable. Please download and unzip the dataset"
        echo "Download Link: https://pan.baidu.com/s/18L-59wgeS96WkObISrytQQ Passwd: bva0"
        echo "The tree of the directory should be:"
        echo "."
        echo "|-- En-Zh"
        echo "|-- test-segment"
        echo "    |-- tst2010"
        echo "    |-- ..."
        echo "|-- train-split"
        echo "    |-- train-segment"
        echo "|-- README.md"

        exit 1
    fi

    # generate manifests
    python3 ${TARGET_DIR}/ted_en_zh/ted_en_zh.py \
    --manifest_prefix="data/manifest" \
    --src-dir="${data_dir}"

    echo "Complete raw data pre-process."
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
    --unit_type "spm" \
    --spm_vocab_size=${nbpe} \
    --spm_mode ${bpemode} \
    --spm_model_prefix ${bpeprefix} \
    --spm_character_coverage 1. \
    --vocab_path="${dict_dir}/vocab.txt" \
    --text_keys 'text' \
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
        --unit_type "spm" \
        --spm_model_prefix ${bpeprefix} \
        --vocab_path="${dict_dir}/vocab.txt" \
        --manifest_path="data/manifest.${set}.raw" \
        --output_path="data/manifest.${set}"

        if [ $? -ne 0 ]; then
            echo "Formt mnaifest failed. Terminated."
            exit 1
        fi
    }&
    done
    wait
fi

echo "Ted En-Zh Data preparation done."
exit 0
