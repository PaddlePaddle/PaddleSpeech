#!/bin/bash

stage=-1
stop_stage=100
dict_dir=data/lang_char

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
bpeprefix="${dict_dir}/bpe_${bpemode}_${nbpe}"

stride_ms=10
window_ms=25
sample_rate=16000
feat_dim=80

source ${MAIN_ROOT}/utils/parse_options.sh


mkdir -p data
mkdir -p ${dict_dir}
TARGET_DIR=${MAIN_ROOT}/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/librispeech/librispeech.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/librispeech" \
    --full_download="True"

    if [ $? -ne 0 ]; then
        echo "Prepare LibriSpeech failed. Terminated."
        exit 1
    fi

    for sub in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
        mv data/manifest.${sub} data/manifest.${sub}.raw
    done

    rm -rf data/manifest.train.raw data/manifest.dev.raw  data/manifest.test.raw
    for sub in train-clean-100 train-clean-360 train-other-500; do
        cat data/manifest.${sub}.raw >> data/manifest.train.raw
    done

    for sub in dev-clean dev-other; do
        cat data/manifest.${sub}.raw >> data/manifest.dev.raw
    done

    for sub in test-clean test-other; do
        cat data/manifest.${sub}.raw >> data/manifest.test.raw
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # compute mean and stddev for normalizer
    num_workers=$(nproc)
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.train.raw" \
    --num_samples=-1 \
    --spectrum_type="fbank" \
    --feat_dim=${feat_dim} \
    --delta_delta=false \
    --sample_rate=${sample_rate} \
    --stride_ms=${stride_ms} \
    --window_ms=${window_ms} \
    --use_dB_normalization=False \
    --num_workers=${num_workers} \
    --output_path="data/mean_std.json"

    if [ $? -ne 0 ]; then
        echo "Compute mean and stddev failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # build vocabulary
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
    --unit_type "spm" \
    --spm_vocab_size=${nbpe} \
    --spm_mode ${bpemode} \
    --spm_model_prefix ${bpeprefix} \
    --vocab_path="${dict_dir}/vocab.txt" \
    --manifest_paths="data/manifest.train.raw"

    if [ $? -ne 0 ]; then
        echo "Build vocabulary failed. Terminated."
        exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for sub in train dev test dev-clean dev-other test-clean test-other; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
        --cmvn_path "data/mean_std.json" \
        --unit_type "spm" \
        --spm_model_prefix ${bpeprefix} \
        --vocab_path="${dict_dir}/vocab.txt" \
        --manifest_path="data/manifest.${sub}.raw" \
        --output_path="data/manifest.${sub}"

        if [ $? -ne 0 ]; then
            echo "Formt mnaifest failed. Terminated."
            exit 1
        fi
    }&
    done
    wait

    for sub in train dev; do
        mv data/manifest.${sub} data/manifest.${sub}.fmt
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for sub in train dev; do
        remove_longshortdata.py --maxframes 3000 --maxchars 400 --stride_ms ${stride_ms} data/manifest.${sub}.fmt data/manifest.${sub}
    done
fi

echo "LibriSpeech Data preparation done."
exit 0
