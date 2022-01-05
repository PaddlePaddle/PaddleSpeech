#!/bin/bash

if [ $# != 4 ];then
    echo "usage: ${0} config_path decode_config_path dict_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
dict_path=$3
ckpt_prefix=$4

batch_size=1
output_dir=${ckpt_prefix}
mkdir -p ${output_dir}

# align dump in `result_file`
# .tier, .TextGrid dump in `dir of result_file`
python3 -u ${BIN_DIR}/test.py \
--model-name 'u2_kaldi' \
--run-mode 'align' \
--dict-path ${dict_path} \
--ngpu ${ngpu} \
--config ${config_path} \
--decode_cfg ${decode_config_path} \
--result-file ${output_dir}/${type}.align \
--checkpoint_path ${ckpt_prefix} \
--opts decode.decode_batch_size ${batch_size}

if [ $? -ne 0 ]; then
    echo "Failed in ctc alignment!"
    exit 1
fi

exit 0
