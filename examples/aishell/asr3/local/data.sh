#!/bin/bash

stage=-1
stop_stage=100
dict_dir=data/lang_char

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

mkdir -p data
mkdir -p ${dict_dir}
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/aishell/aishell.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/aishell"

    if [ $? -ne 0 ]; then
        echo "Prepare Aishell failed. Terminated."
        exit 1
    fi

    for dataset in train dev test; do
        mv data/manifest.${dataset} data/manifest.${dataset}.raw
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # compute mean and stddev for normalizer
    num_workers=$(nproc)
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.train.raw" \
    --spectrum_type="fbank" \
    --feat_dim=80 \
    --delta_delta=false \
    --stride_ms=10 \
    --window_ms=25 \
    --sample_rate=16000 \
    --use_dB_normalization=False \
    --num_samples=-1 \
    --num_workers=${num_workers} \
    --output_path="data/mean_std.json"

    if [ $? -ne 0 ]; then
        echo "Compute mean and stddev failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download data, generate manifests
    # build vocabulary
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
    --unit_type="char" \
    --count_threshold=0 \
    --vocab_path="${dict_dir}/vocab.txt" \
    --manifest_paths "data/manifest.train.raw"

    if [ $? -ne 0 ]; then
        echo "Build vocabulary failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for dataset in train dev test; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
            --cmvn_path "data/mean_std.json" \
            --unit_type "char" \
            --vocab_path="${dict_dir}/vocab.txt" \
            --manifest_path="data/manifest.${dataset}.raw" \
            --output_path="data/manifest.${dataset}"

        if [ $? -ne 0 ]; then
            echo "Formt mnaifest failed. Terminated."
            exit 1
        fi
    } &
    done
    wait
fi
echo "Aishell data preparation done."

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    mkdir -p exp/wav2vec2
    echo "Pretrained wav2vec2 model download"
    wget -P exp/wav2vec2 https://paddlespeech.bj.bcebos.com/wav2vec/chinese-wav2vec2-large.pdparams
fi

exit 0

