IN_MANIFESTS="../datasets/manifest.tmp ../datasets/manifest.dev ../datasets/manifest.test"
OUT_MANIFESTS="./cloud.manifest.tmp ./cloud.manifest.dev ./cloud.manifest.test"
CLOUD_DATA_DIR="/pfs/dlnel/home/sunxinghai@baidu.com/deepspeech2/data"
NUM_SHARDS=10

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
