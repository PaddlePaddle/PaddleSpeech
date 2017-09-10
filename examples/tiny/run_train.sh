#! /usr/bin/bash

pushd ../..

CUDA_VISIBLE_DEVICES=0,1 \
python -u train.py \
--batch_size=2 \
--trainer_count=1 \
--num_passes=10 \
--num_proc_data=1 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--num_iter_print=100 \
--learning_rate=5e-5 \
--max_duration=27.0 \
--min_duration=0.0 \
--use_sortagrad=True \
--use_gru=False \
--use_gpu=True \
--is_local=True \
--share_rnn_weights=True \
--train_manifest='data/tiny/manifest.train' \
--dev_manifest='data/tiny/manifest.train' \
--mean_std_path='data/tiny/mean_std.npz' \
--vocab_path='data/tiny/vocab.txt' \
--output_model_dir='./checkpoints' \
--augment_conf_path='conf/augmentation.config' \
--specgram_type='linear' \
--shuffle_method='batch_shuffle_clipped'
