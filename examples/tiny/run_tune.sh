#! /usr/bin/bash

pushd ../.. > /dev/null

# grid-search for hyper-parameters in language model
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
--tune_manifest='data/tiny/manifest.tiny' \
--mean_std_path='data/tiny/mean_std.npz' \
--vocab_path='data/tiny/vocab.txt' \
--model_path='checkpoints/params.pass-9.tar.gz' \
--lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
--error_rate_type='wer' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in tuning!"
    exit 1
fi


exit 0
