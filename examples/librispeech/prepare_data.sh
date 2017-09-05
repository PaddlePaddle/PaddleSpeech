#! /usr/bin/bash

pushd ../..

# download data, generate manifests
python data/librispeech/librispeech.py \
--manifest_prefix='data/librispeech/manifest' \
--full_download='True' \
--target_dir='~/.cache/paddle/dataset/speech/Libri'

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

#cat data/librispeech/manifest.train* | shuf > data/librispeech/manifest.train


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
