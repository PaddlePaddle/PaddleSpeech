#!/bin/bash

if [ $# != 3 ];then
    echo "usage: ${0} config_path ckpt_path_prefix audio_file"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
ckpt_prefix=$2
audio_file=$3

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
bpeprefix="data/bpe_${bpemode}_${nbpe}"
bpemodel=${bpeprefix}.model

chunk_mode=false
if [[ ${config_path} =~ ^.*chunk_.*yaml$ ]];then
    chunk_mode=true
fi

# download language model
#bash local/download_lm_ch.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi


for type in attention_rescoring; do
    echo "decoding ${type}"
    batch_size=1
    output_dir=${ckpt_prefix}
    mkdir -p ${output_dir}
    python3 -u ${BIN_DIR}/test_hub.py \
    --ngpu ${ngpu} \
    --config ${config_path} \
    --result_file ${output_dir}/${type}.rsl \
    --checkpoint_path ${ckpt_prefix} \
    --opts decoding.decoding_method ${type} \
    --opts decoding.batch_size ${batch_size} \
    --audio_file ${audio_file}

    #score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
done

exit 0
