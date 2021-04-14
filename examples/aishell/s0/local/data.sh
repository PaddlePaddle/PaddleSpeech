#! /usr/bin/env bash

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

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


# build vocabulary
python3 ${MAIN_ROOT}/utils/build_vocab.py \
--unit_type="char" \
--count_threshold=0 \
--vocab_path="data/vocab.txt" \
--manifest_paths "data/manifest.train.raw" "data/manifest.dev.raw"

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
--manifest_path="data/manifest.train.raw" \
--num_samples=2000 \
--specgram_type="fbank" \
--feat_dim=80 \
--delta_delta=false \
--output_path="data/mean_std.npz"

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


# format manifest with tokenids, vocab size
for dataset in train dev test; do
    python3 ${MAIN_ROOT}/utils/format_data.py \
    --feat_type "raw" \
    --cmvn_path "data/mean_std.npz" \
    --unit_type "char" \
    --vocab_path="data/vocab.txt" \
    --manifest_path="data/manifest.${dataset}.raw" \
    --output_path="data/manifest.${dataset}"
done

if [ $? -ne 0 ]; then
    echo "Formt mnaifest failed. Terminated."
    exit 1
fi

echo "Aishell data preparation done."
exit 0
