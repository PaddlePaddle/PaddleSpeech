TRAIN_MANIFEST=$1
DEV_MANIFEST=$2
MODEL_PATH=$3
NUM_GPU=$4
BATCH_SIZE=$5
IS_LOCAL=$6

python ./cloud/split_data.py \
--in_manifest_path=${TRAIN_MANIFEST} \
--out_manifest_path='/local.manifest.train'

python ./cloud/split_data.py \
--in_manifest_path=${DEV_MANIFEST} \
--out_manifest_path='/local.manifest.dev'

python train.py \
--batch_size=$BATCH_SIZE \
--use_gpu=1 \
--trainer_count=${NUM_GPU} \
--num_threads_data=${NUM_GPU} \
--is_local=${IS_LOCAL} \
--train_manifest_path='/local.manifest.train' \
--dev_manifest_path='/local.manifest.dev' \
--output_model_dir=${MODEL_PATH} \
