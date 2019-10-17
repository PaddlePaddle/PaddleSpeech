#! /usr/bin/env bash

cd ../.. > /dev/null

# download language model
cd models/lm > /dev/null
bash download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# download well-trained model
cd models/baidu_en8k > /dev/null
bash download_model.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# infer
CUDA_VISIBLE_DEVICES=0 \
python -u infer.py \
--num_samples=10 \
--beam_size=500 \
--num_proc_bsearch=5 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=1024 \
--alpha=1.4 \
--beta=0.35 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=False \
--share_rnn_weights=False \
--infer_manifest='data/librispeech/manifest.test-clean' \
--mean_std_path='models/baidu_en8k/mean_std.npz' \
--vocab_path='models/baidu_en8k/vocab.txt' \
--model_path='models/baidu_en8k' \
--lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0
