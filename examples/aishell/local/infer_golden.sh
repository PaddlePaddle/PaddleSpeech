#! /usr/bin/env bash

# download language model
bash local/download_lm_ch.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# download well-trained model
bash local/download_model.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# infer
CUDA_VISIBLE_DEVICES=0 \
python3 -u ${MAIN_ROOT}/infer.py \
--num_samples=10 \
--beam_size=300 \
--num_proc_bsearch=8 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=1024 \
--alpha=2.6 \
--beta=5.0 \
--cutoff_prob=0.99 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=False \
--share_rnn_weights=False \
--infer_manifest="data/manifest.test" \
--mean_std_path="data/pretrain/mean_std.npz" \
--vocab_path="data/pretrain/vocab.txt" \
--model_path="data/pretrain" \
--lang_model_path="data/lm/zh_giga.no_cna_cmn.prune01244.klm" \
--decoding_method="ctc_beam_search" \
--error_rate_type="cer" \
--specgram_type="linear"

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0
