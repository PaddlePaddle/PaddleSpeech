#! /usr/bin/env bash

mkdir -p data

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python3 local/librispeech.py \
--manifest_prefix="data/manifest" \
--target_dir="${MAIN_ROOT}/dataset/librispeech" \
--full_download="True"

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

cat data/manifest.train-* | shuf > data/manifest.train


# build vocabulary
python3 ${MAIN_ROOT}/tools/build_vocab.py \
--count_threshold=0 \
--vocab_path="data/vocab.txt" \
--manifest_paths="data/manifest.train"

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python3 ${MAIN_ROOT}/tools/compute_mean_std.py \
--manifest_path="data/manifest.train" \
--num_samples=2000 \
--specgram_type="linear" \
--output_path="data/mean_std.npz"

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "LibriSpeech Data preparation done."
exit 0
