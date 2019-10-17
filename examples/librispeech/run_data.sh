#! /usr/bin/env bash

cd ../.. > /dev/null

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python data/librispeech/librispeech.py \
--manifest_prefix='data/librispeech/manifest' \
--target_dir='./dataset/librispeech' \
--full_download='True'

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

cat data/librispeech/manifest.train-* | shuf > data/librispeech/manifest.train


# build vocabulary
python tools/build_vocab.py \
--count_threshold=0 \
--vocab_path='data/librispeech/vocab.txt' \
--manifest_paths='data/librispeech/manifest.train'

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python tools/compute_mean_std.py \
--manifest_path='data/librispeech/manifest.train' \
--num_samples=2000 \
--specgram_type='linear' \
--output_path='data/librispeech/mean_std.npz'

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


echo "LibriSpeech Data preparation done."
exit 0
