#! /usr/bin/env bash

cd ../.. > /dev/null

# download language model
cd models/lm > /dev/null
bash download_lm_ch.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# evaluate model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u test.py \
--batch_size=128 \
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
--use_gpu=True \
--share_rnn_weights=False \
--test_manifest='data/aishell/manifest.test' \
--mean_std_path='data/aishell/mean_std.npz' \
--vocab_path='data/aishell/vocab.txt' \
--model_path='checkpoints/aishell/step_final' \
--lang_model_path='models/lm/zh_giga.no_cna_cmn.prune01244.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='cer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
