#! /usr/bin/env bash

cd ../.. > /dev/null

# infer
CUDA_VISIBLE_DEVICES=0 \
python -u infer.py \
--num_samples=10 \
--beam_size=300 \
--num_proc_bsearch=8 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=2.6 \
--beta=5.0 \
--cutoff_prob=0.99 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=True \
--share_rnn_weights=False \
--infer_manifest='data/aishell/manifest.test' \
--mean_std_path='models/baidu_cn1.2k/mean_std.npz' \
--vocab_path='models/baidu_cn1.2k/vocab.txt' \
--model_path='models/baidu_cn1.2k' \
--lang_model_path='models/lm/zhidao_giga.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='cer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi


exit 0
