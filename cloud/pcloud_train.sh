DATA_PATH=$1
MODEL_PATH=$2
#setted by user
TRAIN_MANI=${DATA_PATH}/cloud.train.manifest
#setted by user
DEV_MANI=${DATA_PATH}/cloud.test.manifest
#setted by user
TRAIN_TAR=${DATA_PATH}/cloud.train.tar
#setted by user
DEV_TAR=${DATA_PATH}/cloud.test.tar
#setted by user
VOCAB_PATH=${DATA_PATH}/eng_vocab.txt
#setted by user
MEAN_STD_FILE=${DATA_PATH}/mean_std.npz

# split train data for each pcloud node
python ./cloud/split_data.py \
--in_manifest_path=$TRAIN_MANI \
--data_tar_path=$TRAIN_TAR \
--out_manifest_path='./local.train.manifest'

# split dev data for each pcloud node
python ./cloud/split_data.py \
--in_manifest_path=$DEV_MANI \
--data_tar_path=$DEV_TAR \
--out_manifest_path='./local.test.manifest'

python train.py \
--use_gpu=1 \
--trainer_count=4 \
--batch_size=256 \
--mean_std_filepath=$MEAN_STD_FILE \
--train_manifest_path='./local.train.manifest' \
--dev_manifest_path='./local.test.manifest' \
--vocab_filepath=$VOCAB_PATH \
