#!/bin/bash

set -e

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

expdir=exp
datadir=data

recog_set="test-clean test-other dev-clean dev-other"
recog_set="test-clean"

config_path=$1
decode_config_path=$2
ckpt_prefix=$3

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

# download language model
#bash local/download_lm_en.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi

python3 ${MAIN_ROOT}/utils/format_rsl.py \
    --origin_ref data/manifest.test-clean.raw \
    --trans_ref data/manifest.test-clean.text


for type in ctc_greedy_search; do
    echo "decoding ${type}"
    batch_size=16
    python3 -u ${BIN_DIR}/test.py \
        --ngpu ${ngpu} \
        --config ${config_path} \
        --decode_cfg ${decode_config_path} \
        --result_file ${ckpt_prefix}.${type}.rsl \
        --checkpoint_path ${ckpt_prefix} \
        --opts decode.decoding_method ${type} \
        --opts decode.decode_batch_size ${batch_size}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
    python3 ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_hyp ${ckpt_prefix}.${type}.rsl \
        --trans_hyp ${ckpt_prefix}.${type}.rsl.text

    python3 compute_wer.py --char=1 --v=1 \
        data/manifest.test-clean.text ${ckpt_prefix}.${type}.rsl.text > ${ckpt_prefix}.${type}.error
    echo "decoding ${type} done."
done

for type in ctc_prefix_beam_search; do
    echo "decoding ${type}"
    batch_size=1
    python3 -u ${BIN_DIR}/test.py \
        --ngpu ${ngpu} \
        --config ${config_path} \
        --decode_cfg ${decode_config_path} \
        --result_file ${ckpt_prefix}.${type}.rsl \
        --checkpoint_path ${ckpt_prefix} \
        --opts decode.decoding_method ${type} \
        --opts decode.decode_batch_size ${batch_size}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
    python3 ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_hyp ${ckpt_prefix}.${type}.rsl \
        --trans_hyp ${ckpt_prefix}.${type}.rsl.text

    python3 compute_wer.py --char=1 --v=1 \
        data/manifest.test-clean.text ${ckpt_prefix}.${type}.rsl.text > ${ckpt_prefix}.${type}.error
    echo "decoding ${type} done."
done

echo "Finished"

exit 0
