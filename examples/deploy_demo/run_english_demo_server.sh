#! /usr/bin/env bash
# TODO: replace the model with a mandarin model

source path.sh

# download language model
cd ${MAIN_ROOT}/models/lm > /dev/null
bash download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# download well-trained model
cd ${MAIN_ROOT}/models/baidu_en8k > /dev/null
bash download_model.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


# start demo server
CUDA_VISIBLE_DEVICES=0 \
python3 -u ${MAIN_ROOT}/deploy/demo_server.py \
--host_ip="localhost" \
--host_port=8086 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=1024 \
--alpha=1.15 \
--beta=0.15 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=True \
--share_rnn_weights=False \
--speech_save_dir="demo_cache" \
--warmup_manifest="${MAIN_ROOT}/examples/tiny/data/manifest.test-clean" \
--mean_std_path="${MAIN_ROOT}/models/baidu_en8k/mean_std.npz" \
--vocab_path="${MAIN_ROOT}/models/baidu_en8k/vocab.txt" \
--model_path="${MAIN_ROOT}/models/baidu_en8k" \
--lang_model_path="${MAIN_ROOT}/models/lm/common_crawl_00.prune01111.trie.klm" \
--decoding_method="ctc_beam_search" \
--specgram_type="linear"

if [ $? -ne 0 ]; then
    echo "Failed in starting demo server!"
    exit 1
fi


exit 0
