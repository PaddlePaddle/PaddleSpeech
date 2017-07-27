DATA_PATH=/pfs/dlnel/public/dataset/speech/libri
#setted by user
TRAIN_MANI=${DATA_PATH}/manifest_pcloud.train
#setted by user
DEV_MANI=${DATA_PATH}/manifest_pcloud.dev
#setted by user
TRAIN_TAR=${DATA_PATH}/data.train.tar
#setted by user
DEV_TAR=${DATA_PATH}/data.dev.tar
#setted by user
VOCAB_PATH=${DATA_PATH}/eng_vocab.txt
#setted by user
MEAN_STD_FILE=${DATA_PATH}/mean_std.npz

tar -xzf deepspeech.tar.gz
rm -rf ./cloud/data/*

# split train data for each pcloud node
python ./cloud/pcloud_split_data.py \
--in_manifest_path=$TRAIN_MANI \
--data_tar_path=$TRAIN_TAR \
--out_manifest_path='./cloud/data/train.mani'

# split dev data for each pcloud node
python pcloud_split_data.py \
--in_manifest_path=$DEV_MANI \
--data_tar_path=$DEV_TAR \
--out_manifest_path='./cloud/data/dev.mani'

python train.py \
--use_gpu=1 \
--trainer_count=4 \
--batch_size=256 \
--mean_std_filepath=$MEAN_STD_FILE \
--train_manifest_path='./cloud/data/train.mani' \
--dev_manifest_path='./cloud/data/dev.mani' \
--vocab_filepath=$VOCAB_PATH \
