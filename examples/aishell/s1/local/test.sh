#! /usr/bin/env bash

if [ $# != 2 ];then
    echo "usage: ${0} config_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

device=gpu
if [ ngpu == 0 ];then
    device=cpu
fi
config_path=$1
ckpt_prefix=$2

ckpt_name=$(basename ${ckpt_prefxi})

mkdir -p exp

# download language model
#bash local/download_lm_ch.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi


for type in attention ctc_greedy_search; do
    echo "decoding ${type}"
    batch_size=64
    output_dir=${ckpt_prefix}
    mkdir -p ${output_dir}
    python3 -u ${BIN_DIR}/test.py \
    --device ${device} \
    --nproc 1 \
    --config ${config_path} \
    --result_file ${output_dir}/${type}.rsl \
    --checkpoint_path ${ckpt_prefix} \
    --opts decoding.decoding_method ${type} decoding.batch_size ${batch_size}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
done

for type in ctc_prefix_beam_search attention_rescoring; do
    echo "decoding ${type}"
    batch_size=1
    output_dir=${ckpt_prefix}
    mkdir -p ${output_dir}
    python3 -u ${BIN_DIR}/test.py \
    --device ${device} \
    --nproc 1 \
    --config ${config_path} \
    --result_file ${output_dir}/${type}.rsl \
    --checkpoint_path ${ckpt_prefix} \
    --opts decoding.decoding_method ${type} decoding.batch_size ${batch_size}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
done

exit 0
