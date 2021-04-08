#! /usr/bin/env bash

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python3 ${TARGET_DIR}/librispeech/librispeech.py \
--manifest_prefix="data/manifest" \
--target_dir="${TARGET_DIR}/librispeech" \
--full_download="False"

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

head -n 64 data/manifest.dev-clean  > data/manifest.tiny.raw

# bpemode (unigram or bpe)
nbpe=200
bpemode=unigram
bpeprefix="data/bpe_${bpemode}_${nbpe}"
# build vocabulary
python3 ${MAIN_ROOT}/utils/build_vocab.py \
--unit_type "bpe" \
--count_threshold=${nbpe} \
--bpe_mode ${bpemode} \
--bpe_model_prefix ${bpeprefix} \
--vocab_path="data/vocab.txt" \
--manifest_paths="data/manifest.tiny.raw"

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


# compute mean and stddev for normalizer
python3 ${MAIN_ROOT}/utils/compute_mean_std.py \
--manifest_path="data/manifest.tiny.raw" \
--num_samples=64 \
--specgram_type="linear" \
--output_path="data/mean_std.npz"

if [ $? -ne 0 ]; then
    echo "Compute mean and stddev failed. Terminated."
    exit 1
fi


# format manifest with tokenids, vocab size
python3 ${MAIN_ROOT}/utils/format_data.py \
--feat_type "raw" \
--unit_type "bpe" \
--bpe_model_prefix ${bpeprefix} \
--vocab_path="data/vocab.txt" \
--manifest_path="data/manifest.tiny.raw" \
--output_path="data/manifest.tiny"


if [ $? -ne 0 ]; then
    echo "Formt mnaifest failed. Terminated."
    exit 1
fi

echo "LibriSpeech Data preparation done."
exit 0