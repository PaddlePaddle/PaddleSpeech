#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

stage=-1
stop_stage=100

# Use your own data path. You need to download the WenetSpeech dataset by yourself.
wenetspeech_data_dir=./wenetspeech
# Make sure you have 1.2T for ${shards_dir}
shards_dir=./wenetspeech_shards

#wenetspeech training set
set=L
train_set=train_`echo $set | tr 'A-Z' 'a-z'`
dev_set=dev
test_sets="test_net test_meeting"

cmvn=true
cmvn_sampling_divisor=20 # 20 means 5% of the training data to estimate cmvn


. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
set -u
set -o pipefail


mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
    # download data
    echo "Please follow https://github.com/wenet-e2e/WenetSpeech to download the data."
    exit 0;
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Data preparation"
    local/wenetspeech_data_prep.sh \
        --train-subset $set \
        $wenetspeech_data_dir \
        data || exit 1;
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # generate manifests
    python3 ${TARGET_DIR}/aishell/aishell.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/aishell"

    if [ $? -ne 0 ]; then
        echo "Prepare Aishell failed. Terminated."
        exit 1
    fi

    for dataset in train dev test; do
        mv data/manifest.${dataset} data/manifest.${dataset}.raw
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # compute mean and stddev for normalizer
    if $cmvn; then
        full_size=`cat data/${train_set}/wav.scp | wc -l`
        sampling_size=$((full_size / cmvn_sampling_divisor))
        shuf -n $sampling_size data/$train_set/wav.scp \
            > data/$train_set/wav.scp.sampled
        num_workers=$(nproc)

        python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
        --manifest_path="data/manifest.train.raw" \
        --spectrum_type="fbank" \
        --feat_dim=80 \
        --delta_delta=false \
        --stride_ms=10 \
        --window_ms=25 \
        --sample_rate=16000 \
        --use_dB_normalization=False \
        --num_samples=-1 \
        --num_workers=${num_workers} \
        --output_path="data/mean_std.json"

        if [ $? -ne 0 ]; then
            echo "Compute mean and stddev failed. Terminated."
            exit 1
        fi
    fi
fi

dict=data/dict/lang_char.txt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download data, generate manifests
    # build vocabulary
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
    --unit_type="char" \
    --count_threshold=0 \
    --vocab_path="data/vocab.txt" \
    --manifest_paths "data/manifest.train.raw"

    if [ $? -ne 0 ]; then
        echo "Build vocabulary failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for dataset in train dev test; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
            --cmvn_path "data/mean_std.json" \
            --unit_type "char" \
            --vocab_path="data/vocab.txt" \
            --manifest_path="data/manifest.${dataset}.raw" \
            --output_path="data/manifest.${dataset}"

        if [ $? -ne 0 ]; then
            echo "Formt mnaifest failed. Terminated."
            exit 1
        fi
    } &
    done
    wait
fi

echo "Aishell data preparation done."
exit 0
