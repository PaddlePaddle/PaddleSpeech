#!/bin/bash

# ./local/quant.sh conf/chunk_conformer_u2pp.yaml conf/tuning/chunk_decode.yaml exp/chunk_conformer_u2pp/checkpoints/avg_10 data/wav.aishell.test.scp 
if [ $# != 4 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix audio_scp"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3
audio_scp=$4

mkdir -p data
if [ $? -ne 0 ]; then
   exit 1
fi

if [ ! -f ${audio_scp} ]; then
    echo "Plase input the right audio_scp path"
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

for type in  attention_rescoring; do
    echo "decoding ${type}"
    batch_size=1
    output_dir=${ckpt_prefix}
    mkdir -p ${output_dir}
    python3 -u ${BIN_DIR}/quant.py \
    --ngpu ${ngpu} \
    --config ${config_path} \
    --decode_cfg ${decode_config_path} \
    --result_file ${output_dir}/${type}.rsl \
    --checkpoint_path ${ckpt_prefix} \
    --opts decode.decoding_method ${type} \
    --opts decode.decode_batch_size ${batch_size} \
    --num_utts 200 \
    --audio_scp ${audio_scp}

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
done
exit 0
