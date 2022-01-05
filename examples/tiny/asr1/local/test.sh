#!/bin/bash

if [ $# != 3 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3

chunk_mode=false
if [[ ${config_path} =~ ^.*chunk_.*yaml$ ]];then
    chunk_mode=true
fi

# download language model
#bash local/download_lm_en.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi

for type in attention ctc_greedy_search; do
    echo "decoding ${type}"
    if [ ${chunk_mode} == true ];then
        # stream decoding only support batchsize=1
        batch_size=1
    else
        batch_size=64
    fi
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
done

for type in ctc_prefix_beam_search attention_rescoring; do
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
done

exit 0
