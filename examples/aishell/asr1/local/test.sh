#!/bin/bash

set -e

stage=0
stop_stage=100

source utils/parse_options.sh || exit 1;

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."


if [ $# != 3 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix"
    exit -1
fi

config_path=$1
decode_config_path=$2
ckpt_prefix=$3

chunk_mode=false
if [[ ${config_path} =~ ^.*chunk_.*yaml$ ]];then
    chunk_mode=true
fi

# download language model
#bash local/download_lm_ch.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # format the reference test file
    python ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_ref data/manifest.test.raw \
        --trans_ref data/manifest.test.text

    for type in attention ctc_greedy_search; do
        echo "decoding ${type}"
        if [ ${chunk_mode} == true ];then
            # stream decoding only support batchsize=1
            batch_size=1
        else
            batch_size=64
        fi
        output_dir=${ckpt_prefix}
        mkdir -p ${output_dir}
        python3 -u ${BIN_DIR}/test.py \
        --ngpu ${ngpu} \
        --config ${config_path} \
        --decode_cfg ${decode_config_path} \
        --result_file ${output_dir}/${type}.rsl \
        --checkpoint_path ${ckpt_prefix} \
        --opts decode.decoding_method ${type} \
        --opts decode.decode_batch_size ${batch_size}

        if [ $? -ne 0 ]; then
            echo "Failed in evaluation!"
            exit 1

        fi
        # format the hyp file
        python ${MAIN_ROOT}/utils/format_rsl.py \
            --origin_hyp ${output_dir}/${type}.rsl \
            --trans_hyp ${output_dir}/${type}.rsl.text
        python ${MAIN_ROOT}/utils/compute-wer.py --char=1 --v=1 \
            data/manifest.test.text ${output_dir}/${type}.rsl.text > ${output_dir}/${type}.error 

    done

    for type in ctc_prefix_beam_search attention_rescoring; do
        echo "decoding ${type}"
        batch_size=1
        output_dir=${ckpt_prefix}
        mkdir -p ${output_dir}
        python3 -u ${BIN_DIR}/test.py \
        --ngpu ${ngpu} \
        --config ${config_path} \
        --decode_cfg ${decode_config_path} \
        --result_file ${output_dir}/${type}.rsl \
        --checkpoint_path ${ckpt_prefix} \
        --opts decode.decoding_method ${type} \
        --opts decode.decode_batch_size ${batch_size}

        if [ $? -ne 0 ]; then
            echo "Failed in evaluation!"
            exit 1
        fi
        python ${MAIN_ROOT}/utils/format_rsl.py \
            --origin_hyp ${output_dir}/${type}.rsl \
            --trans_hyp ${output_dir}/${type}.rsl.text
        python ${MAIN_ROOT}/utils/compute-wer.py --char=1 --v=1 \
            data/manifest.test.text ${output_dir}/${type}.rsl.text > ${output_dir}/${type}.error 
    done
fi

if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
    echo "using sclite to compute cer..."
    # format the reference test file for sclite
    python ${MAIN_ROOT}/utils/format_rsl.py \
        --origin_ref data/manifest.test.raw \
        --trans_ref_sclite data/manifest.test.text.sclite
    
    output_dir=${ckpt_prefix}
    for type in attention ctc_greedy_search ctc_prefix_beam_search attention_rescoring; do
        python ${MAIN_ROOT}/utils/format_rsl.py \
            --origin_hyp ${output_dir}/${type}.rsl \
            --trans_hyp_sclite ${output_dir}/${type}.rsl.text.sclite

        mkdir -p ${output_dir}/${type}_sclite
        sclite -i wsj -r data/manifest.test.text.sclite -h  ${output_dir}/${type}.rsl.text.sclite  -e utf-8 -o all -O ${output_dir}/${type}_sclite -c NOASCII
    done
fi

exit 0
