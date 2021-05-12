#! /usr/bin/env bash

stage=-1
stop_stage=100

# bpemode (unigram or bpe)
nbpe=200
bpemode=unigram
bpeprefix="data/bpe_${bpemode}_${nbpe}"

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/librispeech/librispeech.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/librispeech" \
    --full_download="False"
    
    if [ $? -ne 0 ]; then
        echo "Prepare LibriSpeech failed. Terminated."
        exit 1
    fi
    
    head -n 64 data/manifest.dev-clean  > data/manifest.tiny.raw
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # build vocabulary
    python3 ${MAIN_ROOT}/utils/build_vocab.py \
    --unit_type "spm" \
    --spm_vocab_size=${nbpe} \
    --spm_mode ${bpemode} \
    --spm_model_prefix ${bpeprefix} \
    --vocab_path="data/vocab.txt" \
    --manifest_paths="data/manifest.tiny.raw"
    
    if [ $? -ne 0 ]; then
        echo "Build vocabulary failed. Terminated."
        exit 1
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # compute mean and stddev for normalizer
    python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
    --manifest_path="data/manifest.tiny.raw" \
    --num_samples=64 \
    --specgram_type="fbank" \
    --feat_dim=80 \
    --delta_delta=false \
    --sample_rate=16000 \
    --stride_ms=10.0 \
    --window_ms=25.0 \
    --use_dB_normalization=False \
    --num_workers=2 \
    --output_path="data/mean_std.json"
    
    if [ $? -ne 0 ]; then
        echo "Compute mean and stddev failed. Terminated."
        exit 1
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    python3 ${MAIN_ROOT}/utils/format_data.py \
    --feat_type "raw" \
    --cmvn_path "data/mean_std.json" \
    --unit_type "spm" \
    --spm_model_prefix ${bpeprefix} \
    --vocab_path="data/vocab.txt" \
    --manifest_path="data/manifest.tiny.raw" \
    --output_path="data/manifest.tiny"
    
    
    if [ $? -ne 0 ]; then
        echo "Formt mnaifest failed. Terminated."
        exit 1
    fi
fi

echo "LibriSpeech Data preparation done."
exit 0
