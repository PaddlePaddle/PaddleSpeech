#! /usr/bin/env bash

# download data, generate manifests
PYTHONPATH=../../:$PYTHONPATH python voxforge.py \
--manifest_prefix='./manifest' \
--target_dir='./dataset/VoxForge' \
--is_merge_dialect=True \
--dialects 'american' 'british' 'australian' 'european' 'irish' 'canadian' 'indian'

if [ $? -ne 0 ]; then
    echo "Prepare VoxForge failed. Terminated."
    exit 1
fi

echo "VoxForge Data preparation done."
exit 0
