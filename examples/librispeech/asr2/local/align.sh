#!/bin/bash

if [ $# != 3 ];then
    echo "usage: ${0} config_path dict_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
dict_path=$2
ckpt_prefix=$3

batch_size=1
output_dir=${ckpt_prefix}
mkdir -p ${output_dir}

# align dump in `result_file`
# .tier, .TextGrid dump in `dir of result_file`
python3 -u ${BIN_DIR}/test.py \
--model-name 'u2_kaldi' \
--run-mode 'align' \
--dict-path ${dict_path} \
--nproc ${ngpu} \
--config ${config_path} \
--result-file ${output_dir}/${type}.align \
--checkpoint_path ${ckpt_prefix} \
--opts decoding.batch_size ${batch_size}

if [ $? -ne 0 ]; then
    echo "Failed in ctc alignment!"
    exit 1
fi

exit 0
