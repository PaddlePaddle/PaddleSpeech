#! /usr/bin/bash

pushd ../..

# download data, generate manifests
python data/tiny/tiny.py \
--manifest_prefix='data/tiny/manifest' \
--target_dir=$HOME'/.cache/paddle/dataset/speech/tiny'

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

cat data/tiny/manifest.dev-clean | head -n 32 > data/tiny/manifest.train
cat data/tiny/manifest.dev-clean | head -n 48 | tail -n 16 > data/tiny/manifest.dev
cat data/tiny/manifest.dev-clean | head -n 64 | tail -n 16 > data/tiny/manifest.test


# build vocabulary
python tools/build_vocab.py \
--count_threshold=0 \
--vocab_path='data/tiny/vocab.txt' \
--manifest_paths='data/tiny/manifest.train'

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python tools/compute_mean_std.py \
--manifest_path='data/tiny/manifest.train' \
--num_samples=32 \
--specgram_type='linear' \
--output_path='data/tiny/mean_std.npz'

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "Tiny data preparation done."
