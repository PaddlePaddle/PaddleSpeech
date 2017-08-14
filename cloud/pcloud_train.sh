DATA_PATH=$1
MODEL_PATH=$2
NUM_CPU=$3
NUM_GPU=$4
IS_LOCAL=$5

TRAIN_MANI=${DATA_PATH}/cloud.train.manifest
DEV_MANI=${DATA_PATH}/cloud.dev.manifest
TRAIN_TAR=${DATA_PATH}/cloud.train.tar
DEV_TAR=${DATA_PATH}/cloud.dev.tar
VOCAB_PATH=${DATA_PATH}/vocab.txt
MEAN_STD_FILE=${DATA_PATH}/mean_std.npz

# split train data for each pcloud node
python ./cloud/split_data.py \
--in_manifest_path=${TRAIN_MANI} \
--data_tar_path=${TRAIN_TAR} \
--out_manifest_path='/local.train.manifest'

# split dev data for each pcloud node
python ./cloud/split_data.py \
--in_manifest_path=${DEV_MANI} \
--data_tar_path=${DEV_TAR} \
--out_manifest_path='/local.dev.manifest'

# run train
python train.py \
--use_gpu=1 \
--trainer_count=${NUM_GPU} \
--num_threads_data=${NUM_CPU} \
--is_local=${IS_LOCAL} \
--mean_std_filepath=${MEAN_STD_FILE} \
--train_manifest_path='/local.train.manifest' \
--dev_manifest_path='/local.dev.manifest' \
--vocab_filepath=${VOCAB_PATH} \
--output_model_dir=${MODEL_PATH}
