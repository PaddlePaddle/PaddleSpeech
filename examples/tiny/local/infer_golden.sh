#! /usr/bin/env bash

# download language model
cd ${MAIN_ROOT}/models/lm > /dev/null
bash download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# download well-trained model
cd ${MAIN_ROOT}/models/librispeech > /dev/null
bash download_model.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# infer
CUDA_VISIBLE_DEVICES=0 \
python3 -u ${MAIN_ROOT}/infer.py \
--num_samples=10 \
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
--infer_manifest="data/manifest.test-clean" \
--mean_std_path="${MAIN_ROOT}/models/librispeech/mean_std.npz" \
--vocab_path="${MAIN_ROOT}/models/librispeech/vocab.txt" \
--model_path="${MAIN_ROOT}/models/librispeech" \
--lang_model_path="${MAIN_ROOT}/models/lm/common_crawl_00.prune01111.trie.klm" \
--decoding_method="ctc_beam_search" \
--error_rate_type="wer" \
--specgram_type="linear"

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0
