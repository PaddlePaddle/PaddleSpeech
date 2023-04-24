#!/bin/bash

if [ $# != 4 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix audio_file"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3
audio_file=$4

mkdir -p data
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/en/demo_002_en.wav -P data/
if [ $? -ne 0 ]; then
   exit 1
fi

if [ ! -f ${audio_file} ]; then
    echo "Plase input the right audio_file path"
    exit 1
fi

chunk_mode=false
if [[ ${config_path} =~ ^.*chunk_.*yaml$ ]];then
    chunk_mode=true
fi

# download language model
#bash local/download_lm_ch.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi

for type in ctc_greedy_search; do
    echo "decoding ${type}"
    batch_size=1
    output_dir=${ckpt_prefix}
    mkdir -p ${output_dir}
    python3 -u ${BIN_DIR}/test_wav.py \
    --ngpu ${ngpu} \
    --config ${config_path} \
    --decode_cfg ${decode_config_path} \
    --result_file ${output_dir}/${type}.rsl \
    --checkpoint_path ${ckpt_prefix} \
    --opts decode.decoding_method ${type} \
    --opts decode.decode_batch_size ${batch_size} \
    --audio_file ${audio_file}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
done
exit 0
