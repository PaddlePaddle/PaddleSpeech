#
TRAIN_MANIFEST="/home/work/wanghaoshuang/ds2/pcloud/models/deep_speech_2/datasets/manifest.dev"
TEST_MANIFEST="/home/work/wanghaoshuang/ds2/pcloud/models/deep_speech_2/datasets/manifest.dev"
VOCAB_PATH="/home/work/wanghaoshuang/ds2/pcloud/models/deep_speech_2/datasets/vocab/eng_vocab.txt"
MEAN_STD_PATH="/home/work/wanghaoshuang/ds2/pcloud/models/deep_speech_2/compute_mean_std.py"
CLOUD_DATA_DIR="/pfs/dlnel/home/wanghaoshuang@baidu.com/deepspeech2/data"
CLOUD_MODEL_DIR="/pfs/dlnel/home/wanghaoshuang@baidu.com/deepspeech2/model"

DS2_PATH=${PWD%/*}

rm -rf ./tmp
mkdir ./tmp

paddlecloud ls ${CLOUD_DATA_DIR}/mean_std.npz
if [ $? -ne 0 ];then
    cp -f  ${MEAN_STD_PATH} ./tmp/mean_std.npz
    paddlecloud file put ./tmp/mean_std.npz  ${CLOUD_DATA_DIR}/
fi

paddlecloud ls ${CLOUD_DATA_DIR}/vocab.txt
if [ $? -ne 0 ];then
    cp -f  ${VOCAB_PATH} ./tmp/vocab.txt
    paddlecloud file put ./tmp/vocab.txt  ${CLOUD_DATA_DIR}/
fi

paddlecloud ls ${CLOUD_DATA_DIR}/cloud.train.manifest
if [ $? -ne 0 ];then
python prepare_data.py \
--manifest_path=${TRAIN_MANIFEST} \
--out_tar_path="./tmp/cloud.train.tar" \
--out_manifest_path="tmp/cloud.train.manifest"
paddlecloud file put ./tmp/cloud.train.tar ${CLOUD_DATA_DIR}/
paddlecloud file put ./tmp/cloud.train.manifest ${CLOUD_DATA_DIR}/
fi

paddlecloud ls ${CLOUD_DATA_DIR}/cloud.test.manifest
if [ $? -ne 0 ];then
python prepare_data.py \
--manifest_path=${TEST_MANIFEST} \
--out_tar_path="./tmp/cloud.test.tar" \
--out_manifest_path="tmp/cloud.test.manifest"
paddlecloud file put ./tmp/cloud.test.tar ${CLOUD_DATA_DIR}/
paddlecloud file put ./tmp/cloud.test.manifest ${CLOUD_DATA_DIR}/
fi

rm -rf ./tmp

JOB_NAME=deepspeech`date +%Y%m%d%H%M%S`
cp pcloud_train.sh ${DS2_PATH}
paddlecloud submit \
-image bootstrapper:5000/wanghaoshuang/pcloud_ds2:latest-gpu-cudnn \
-jobname ${JOB_NAME} \
-cpu 4 \
-gpu 4 \
-memory 10Gi \
-parallelism 1 \
-pscpu 1 \
-pservers 1 \
-psmemory 10Gi \
-passes 1 \
-entry "sh pcloud_train.sh ${CLOUD_DATA_DIR} ${CLOUD_MODEl_DIR}" \
${DS2_PATH}
