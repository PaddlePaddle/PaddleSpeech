#! /usr/bin/env bash

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

python -u train.py \
--batch_size=${BATCH_SIZE} \
--trainer_count=${NUM_GPU} \
--num_passes=200 \
--num_proc_data=${NUM_GPU} \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--num_iter_print=100 \
--learning_rate=5e-4 \
--max_duration=27.0 \
--min_duration=0.0 \
--use_sortagrad=True \
--use_gru=False \
--use_gpu=True \
--is_local=${IS_LOCAL} \
--share_rnn_weights=True \
--train_manifest='/local.manifest.train' \
--dev_manifest='/local.manifest.dev' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/eng_vocab.txt' \
--output_model_dir='./checkpoints' \
--output_model_dir=${MODEL_PATH} \
--augment_conf_path='conf/augmentation.config' \
--specgram_type='linear' \
--shuffle_method='batch_shuffle_clipped' \
2>&1 | tee ./log/train.log
