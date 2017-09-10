#! /usr/bin/bash

pushd ../..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u tools/tune.py \
--num_samples=100 \
--trainer_count=8 \
--beam_size=500 \
--num_proc_bsearch=12 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--num_alphas=14 \
--num_betas=20 \
--alpha_from=0.1 \
--alpha_to=0.36 \
--beta_from=0.05 \
--beta_to=1.0 \
--cutoff_prob=0.99 \
--use_gru=False \
--use_gpu=True \
--share_rnn_weights=True \
--tune_manifest='data/librispeech/manifest.dev-clean' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/eng_vocab.txt' \
--model_path='checkpoints/params.latest.tar.gz' \
--lang_model_path='lm/data/common_crawl_00.prune01111.trie.klm' \
--error_rate_type='wer' \
--specgram_type='linear'
