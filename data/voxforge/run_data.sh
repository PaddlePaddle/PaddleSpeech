#! /usr/bin/env bash

cd ../.. > /dev/null

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python data/voxforge/voxforge.py \
--manifest_prefix='data/voxforge/manifest' \
--target_dir='~/.cache/paddle/dataset/speech/VoxForge' \
--is_merge_dialect=True \
--dialects 'american' 'british' 'australian' 'european' 'irish' 'canadian' 'indian'

if [ $? -ne 0 ]; then
    echo "Prepare VoxForge failed. Terminated."
    exit 1
fi

echo "VoxForge Data preparation done."
exit 0
