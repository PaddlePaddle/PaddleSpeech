IN_MANIFESTS="../datasets/manifest.train ../datasets/manifest.dev ../datasets/manifest.test"
OUT_MANIFESTS="./cloud.manifest.train ./cloud.manifest.dev ./cloud.manifest.test"
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
