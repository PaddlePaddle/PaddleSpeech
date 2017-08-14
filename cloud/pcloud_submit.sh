# Configure input data set in local filesystem
TRAIN_MANIFEST="../datasets/manifest.train"
DEV_MANIFEST="../datasets/manifest.dev"
VOCAB_FILE="../datasets/vocab/eng_vocab.txt"
MEAN_STD_FILE="../mean_std.npz"
# Configure output path in PaddleCloud filesystem
CLOUD_DATA_DIR="/pfs/dlnel/home/sunxinghai@baidu.com/deepspeech2/data"
CLOUD_MODEL_DIR="/pfs/dlnel/home/sunxinghai@baidu.com/deepspeech2/model"
# Configure cloud resources
NUM_CPU=12
NUM_GPU=8
NUM_NODE=1
MEMORY="10Gi"
IS_LOCAL="True"

# Pack and upload local data to PaddleCloud filesystem
python upload_data.py \
--train_manifest_path=${TRAIN_MANIFEST} \
--dev_manifest_path=${DEV_MANIFEST} \
--vocab_file=${VOCAB_FILE} \
--mean_std_file=${MEAN_STD_FILE} \
--cloud_data_path=${CLOUD_DATA_DIR}
if [ $? -ne 0 ]
then
    echo "upload data failed!"
    exit 1
fi

# Submit job to PaddleCloud
JOB_NAME=deepspeech-`date +%Y%m%d%H%M%S`
DS2_PATH=${PWD%/*}
cp -f  pcloud_train.sh ${DS2_PATH}

paddlecloud submit \
-image bootstrapper:5000/wanghaoshuang/pcloud_ds2:latest \
-jobname ${JOB_NAME} \
-cpu ${NUM_CPU} \
-gpu ${NUM_GPU} \
-memory ${MEMORY} \
-parallelism ${NUM_NODE} \
-pscpu 1 \
-pservers 1 \
-psmemory ${MEMORY} \
-passes 1 \
-entry "sh pcloud_train.sh ${CLOUD_DATA_DIR} ${CLOUD_MODEL_DIR} ${NUM_CPU} ${NUM_GPU} ${IS_LOCAL}" \
${DS2_PATH}

rm ${DS2_PATH}/pcloud_train.sh
