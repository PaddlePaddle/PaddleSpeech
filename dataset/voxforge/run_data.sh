#! /usr/bin/env bash

TARGET_DIR=${MAIN_ROOT}/dataset/voxforge
mkdir -p ${TARGET_DIR}

# download data, generate manifests
python ${MAIN_ROOT}/dataset/voxforge/voxforge.py \
--manifest_prefix="${TARGET_DIR}/manifest" \
--target_dir="${TARGET_DIR}" \
--is_merge_dialect=True \
--dialects 'american' 'british' 'australian' 'european' 'irish' 'canadian' 'indian'

if [ $? -ne 0 ]; then
    echo "Prepare VoxForge failed. Terminated."
    exit 1
fi

echo "VoxForge Data preparation done."
exit 0
