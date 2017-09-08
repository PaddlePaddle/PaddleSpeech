#! /usr/bin/bash

pushd ../..

CUDA_VISIBLE_DEVICES=0 \
python -u infer.py \
--num_samples=10 \
--trainer_count=1 \
--beam_size=500 \
--num_proc_bsearch=12 \
--num_proc_data=12 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=2.15 \
--beta=0.35 \
--cutoff_prob=1.0 \
--use_gru=False \
--use_gpu=True \
--share_rnn_weights=True \
--infer_manifest='data/librispeech/manifest.dev-clean' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/eng_vocab.txt' \
--model_path='checkpoints/params.latest.tar.gz' \
--lang_model_path='lm/data/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--specgram_type='linear'
