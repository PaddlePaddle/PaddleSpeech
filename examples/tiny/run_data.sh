#! /usr/bin/env bash

cd ../.. > /dev/null

# prepare folder
if [ ! -e data/tiny ]; then
    mkdir data/tiny
fi

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python data/librispeech/librispeech.py \
--manifest_prefix='data/tiny/manifest' \
--target_dir='./dataset/librispeech' \
--full_download='False'

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

head -n 64 data/tiny/manifest.dev-clean  > data/tiny/manifest.tiny

# build vocabulary
python tools/build_vocab.py \
--count_threshold=0 \
--vocab_path='data/tiny/vocab.txt' \
--manifest_paths='data/tiny/manifest.tiny'

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python tools/compute_mean_std.py \
--manifest_path='data/tiny/manifest.tiny' \
--num_samples=64 \
--specgram_type='linear' \
--output_path='data/tiny/mean_std.npz'

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "LibriSpeech Data preparation done."
exit 0
