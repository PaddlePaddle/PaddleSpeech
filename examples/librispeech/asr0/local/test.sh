#!/bin/bash

if [ $# != 3 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix"
    exit -1
fi
stage=0
stop_stage=100

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3

# download language model
bash local/download_lm_en.sh
if [ $? -ne 0 ]; then
   exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # format the reference test file
    python3 ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_ref data/manifest.test-clean.raw \
        --trans_ref data/manifest.test-clean.text

    python3 -u ${BIN_DIR}/test.py \
    --ngpu ${ngpu} \
    --config ${config_path} \
    --decode_cfg ${decode_config_path} \
    --result_file ${ckpt_prefix}.rsl \
    --checkpoint_path ${ckpt_prefix}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi

    python3 ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_hyp ${ckpt_prefix}.rsl \
        --trans_hyp ${ckpt_prefix}.rsl.text

    python3 ${MAIN_ROOT}/utils/compute-wer.py --char=1 --v=1 \
        data/manifest.test-clean.text ${ckpt_prefix}.rsl.text > ${ckpt_prefix}.error
fi

if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
    python3 ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_ref data/manifest.test-clean.raw \
        --trans_ref_sclite data/manifest.test.text-clean.sclite

    python3 ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_hyp ${ckpt_prefix}.rsl \
        --trans_hyp_sclite ${ckpt_prefix}.rsl.text.sclite

    mkdir -p ${ckpt_prefix}_sclite
    sclite -i wsj -r data/manifest.test-clean.text.sclite -h  ${ckpt_prefix}.rsl.text.sclite  -e utf-8 -o all -O ${ckpt_prefix}_sclite -c NOASCII
fi


exit 0
