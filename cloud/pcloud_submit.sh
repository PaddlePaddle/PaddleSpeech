# Configure input data set in local filesystem
TRAIN_MANIFEST="/home/work/demo/ds2/pcloud/models/deep_speech_2/datasets/manifest.dev"
TEST_MANIFEST="/home/work/demo/ds2/pcloud/models/deep_speech_2/datasets/manifest.dev"
VOCAB_FILE="/home/work/demo/ds2/pcloud/models/deep_speech_2/datasets/vocab/eng_vocab.txt"
MEAN_STD_FILE="/home/work/demo/ds2/pcloud/models/deep_speech_2/mean_std.npz"

# Configure output path in PaddleCloud filesystem
CLOUD_DATA_DIR="/pfs/dlnel/home/demo/deepspeech2/data"
CLOUD_MODEL_DIR="/pfs/dlnel/home/demo/deepspeech2/model"

# Pack and upload local data to PaddleCloud filesystem
python upload_data.py \
--train_manifest_path=${TRAIN_MANIFEST} \
--test_manifest_path=${TEST_MANIFEST} \
--vocab_file=${VOCAB_FILE} \
--mean_std_file=${MEAN_STD_FILE} \
--cloud_data_path=${CLOUD_DATA_DIR}

JOB_NAME=deepspeech`date +%Y%m%d%H%M%S`
DS2_PATH=${PWD%/*}
cp -f  pcloud_train.sh ${DS2_PATH}

# Configure computation resource and submit job to PaddleCloud
paddlecloud submit \
-image bootstrapper:5000/wanghaoshuang/pcloud_ds2:latest \
-jobname ${JOB_NAME} \
-cpu 4 \
-gpu 4 \
-memory 10Gi \
-parallelism 1 \
-pscpu 1 \
-pservers 1 \
-psmemory 10Gi \
-passes 1 \
-entry "sh pcloud_train.sh ${CLOUD_DATA_DIR} ${CLOUD_MODEL_DIR}" \
${DS2_PATH}

rm ${DS2_PATH}/pcloud_train.sh
