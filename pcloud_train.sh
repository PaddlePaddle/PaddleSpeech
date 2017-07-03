#setted by user
TRAIN_MANI='/pfs/dlnel/home/yanxu05@baidu.com/wanghaoshuang/data/ds2_data/demo.mani'
#setted by user
DEV_MANI='/pfs/dlnel/home/yanxu05@baidu.com/wanghaoshuang/data/ds2_data/demo.mani'
#setted by user
TRAIN_TAR='/pfs/dlnel/home/yanxu05@baidu.com/wanghaoshuang/data/ds2_data/demo.tar'
#setted by user
DEV_TAR='/pfs/dlnel/home/yanxu05@baidu.com/wanghaoshuang/data/ds2_data/demo.tar'
#setted by user
VOCAB_PATH='/pfs/dlnel/home/yanxu05@baidu.com/wanghaoshuang/data/ds2_data/eng_vocab.txt'
#setted by user
MEAN_STD_FILE='/pfs/dlnel/home/yanxu05@baidu.com/wanghaoshuang/data/ds2_data/mean_std.npz'

# split train data for each pcloud node
python pcloud_split_data.py \
--in_manifest_path=$TRAIN_MANI \
--data_tar_path=$TRAIN_TAR \
--out_manifest_path='./train.mani'
# split dev data for each pcloud node
python pcloud_split_data.py \
--in_manifest_path=$DEV_MANI \
--data_tar_path=$DEV_TAR \
--out_manifest_path='./dev.mani'

python train.py \
--use_gpu=0 \
--trainer_count=4 \
--batch_size=2 \
--mean_std_filepath=$MEAN_STD_FILE \
--train_manifest_path='./train.mani' \
--dev_manifest_path='./dev.mani' \
--vocab_filepath=$VOCAB_PATH \
