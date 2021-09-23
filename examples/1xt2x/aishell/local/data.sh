#!/bin/bash

stage=-1
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

bash local/download_model.sh
if [ $? -ne 0 ]; then
   exit 1
fi

tar xzvf aishell_model_v1.8_to_v2.x.tar.gz
mv aishell_v1.8.pdparams exp/deepspeech2/checkpoints/
mv README.md exp/deepspeech2/
mv mean_std.npz data/
mv vocab.txt data/
rm aishell_model_v1.8_to_v2.x.tar.gz -f


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


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # compute mean and stddev for normalizer
    num_workers=$(nproc)
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.train.raw" \
    --specgram_type="linear" \
    --delta_delta=false \
    --stride_ms=10.0 \
    --window_ms=20.0 \
    --sample_rate=16000 \
    --use_dB_normalization=True \
    --num_samples=2000 \
    --num_workers=${num_workers} \
    --output_path="data/mean_std.json"

    if [ $? -ne 0 ]; then
        echo "Compute mean and stddev failed. Terminated."
        exit 1
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for dataset in train dev test; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
                --feat_type "raw" \
                --cmvn_path "data/mean_std.json" \
                --unit_type "char" \
                --vocab_path="data/vocab.txt" \
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
exit 0
