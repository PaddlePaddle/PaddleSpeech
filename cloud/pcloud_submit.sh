TRAIN_MANIFEST="cloud/cloud.manifest.train"
DEV_MANIFEST="cloud/cloud.manifest.dev"
CLOUD_MODEL_DIR="/pfs/dlnel/home/USERNAME/deepspeech2/model"
BATCH_SIZE=256
NUM_GPU=8
NUM_NODE=1
IS_LOCAL="True"

JOB_NAME=deepspeech-`date +%Y%m%d%H%M%S`
DS2_PATH=${PWD%/*}
cp -f  pcloud_train.sh ${DS2_PATH}

paddlecloud submit \
-image bootstrapper:5000/wanghaoshuang/pcloud_ds2:latest \
-jobname ${JOB_NAME} \
-cpu ${NUM_GPU} \
-gpu ${NUM_GPU} \
-memory 64Gi \
-parallelism ${NUM_NODE} \
-pscpu 1 \
-pservers 1 \
-psmemory 64Gi \
-passes 1 \
-entry "sh pcloud_train.sh ${TRAIN_MANIFEST} ${DEV_MANIFEST} ${CLOUD_MODEL_DIR} ${NUM_GPU} ${BATCH_SIZE} ${IS_LOCAL}" \
${DS2_PATH}

rm ${DS2_PATH}/pcloud_train.sh
