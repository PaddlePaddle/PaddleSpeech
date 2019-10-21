#! /usr/bin/env bash

cd ../.. > /dev/null

# download language model
cd models/lm > /dev/null
bash download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# evaluate model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u test.py \
--batch_size=128 \
--beam_size=500 \
--num_proc_bsearch=8 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=2.5 \
--beta=0.3 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=False \
--use_gpu=True \
--share_rnn_weights=True \
--test_manifest='data/librispeech/manifest.test-clean' \
--mean_std_path='data/librispeech/mean_std.npz' \
--vocab_path='data/librispeech/vocab.txt' \
--model_path='checkpoints/libri/step_final' \
--lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
