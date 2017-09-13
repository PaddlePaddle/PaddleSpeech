#! /usr/bin/bash
# TODO: replace the model with a mandarin model

pushd ../.. > /dev/null

# download language model
pushd models/lm > /dev/null
sh download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
popd > /dev/null


# download well-trained model
pushd models/librispeech > /dev/null
sh download_model.sh
if [ $? -ne 0 ]; then
    exit 1
fi
popd > /dev/null


# start demo server
CUDA_VISIBLE_DEVICES=0 \
python -u deploy/demo_server.py \
--host_ip='localhost' \
--host_port=8086 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=0.36 \
--beta=0.25 \
--cutoff_prob=0.99 \
--use_gru=False \
--use_gpu=True \
--share_rnn_weights=True \
--speech_save_dir='demo_cache' \
--warmup_manifest='data/tiny/manifest.test-clean' \
--mean_std_path='models/librispeech/mean_std.npz' \
--vocab_path='models/librispeech/vocab.txt' \
--model_path='models/librispeech/params.tar.gz' \
--lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in starting demo server!"
    exit 1
fi


exit 0
