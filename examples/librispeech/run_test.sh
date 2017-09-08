#! /usr/bin/bash

pushd ../..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u test.py \
--batch_size=128 \
--trainer_count=8 \
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
--test_manifest='data/librispeech/manifest.test-clean' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/eng_vocab.txt' \
--model_path='checkpoints/params.latest.tar.gz' \
--lang_model_path='lm/data/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--specgram_type='linear'
