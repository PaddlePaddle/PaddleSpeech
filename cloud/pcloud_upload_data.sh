#! /usr/bin/env bash

mkdir cloud_manifests

IN_MANIFESTS="../data/librispeech/manifest.train ../data/librispeech/manifest.dev-clean ../data/librispeech/manifest.test-clean"
OUT_MANIFESTS="cloud_manifests/cloud.manifest.train cloud_manifests/cloud.manifest.dev cloud_manifests/cloud.manifest.test"
CLOUD_DATA_DIR="/pfs/dlnel/home/USERNAME/deepspeech2/data/librispeech"
NUM_SHARDS=50

python upload_data.py \
--in_manifest_paths ${IN_MANIFESTS} \
--out_manifest_paths ${OUT_MANIFESTS} \
--cloud_data_dir ${CLOUD_DATA_DIR} \
--num_shards ${NUM_SHARDS}

if [ $? -ne 0 ]
then
    echo "Upload Data Failed!"
    exit 1
fi

echo "All Done."
