stage=-1
stop_stage=100
TARGET_DIR=${MAIN_ROOT}/dataset

. utils/parse_options.sh || exit -1;

src=$1
mkdir -p data/{dev,test}
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    # create data/{dev,test} directory to store the manifest files
    /home/users/xiongxinlei/.conda/envs/xxl_base/bin/python3 ${TARGET_DIR}/voxceleb/voxceleb1.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${src}"

    if [ $? -ne 0 ]; then
        echo "Prepare Voxceleb failed. Terminated."
        exit 1
    fi
    mv data/manifest.dev data/dev
    mv data/voxceleb1.dev.meta data/dev

    mv data/manifest.test data/test
    mv data/voxceleb1.test.meta data/test
fi