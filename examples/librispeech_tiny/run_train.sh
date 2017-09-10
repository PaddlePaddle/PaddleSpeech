#! /usr/bin/bash

pushd ../..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u train.py \
--batch_size=256 \
--trainer_count=8 \
--num_passes=50 \
--num_proc_data=12 \
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
--is_local=True \
--share_rnn_weights=True \
--train_manifest='data/librispeech/manifest.train' \
--dev_manifest='data/librispeech/manifest.dev' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/eng_vocab.txt' \
--output_model_dir='./checkpoints' \
--augment_conf_path='conf/augmentation.config' \
--specgram_type='linear' \
--shuffle_method='batch_shuffle_clipped'
