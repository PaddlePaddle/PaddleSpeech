#! /usr/bin/env bash

if [ $# != 4 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix lang"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3
tgt_lang=$4

for type in fullsentence; do
    echo "decoding ${type}"
    python3 -u ${BIN_DIR}/test.py \
    --ngpu ${ngpu} \
    --config ${config_path} \
    --decode_cfg ${decode_config_path} \
    --result_file ${ckpt_prefix}.${type}.rsl \
    --checkpoint_path ${ckpt_prefix} \
    --opts decode.decoding_method ${type} \

    if [ $? -ne 0 ]; then
        echo "Failed in evaluation!"
        exit 1
    fi
    echo $PATH
    python3 ${MAIN_ROOT}/utils/rsl2trn.py --rsl ${ckpt_prefix}.${type}.rsl \
                            --hyp ${ckpt_prefix}.${type}.hyp \
                            --ref ${ckpt_prefix}.${type}.ref
    if ! which tokenizer.perl > /dev/null; then
    echo "Error: it seems that moses is not installed." >&2
    echo "Error: please install moses as follows." >&2
    echo "Error: cd ${MAIN_ROOT}/tools && make moses.done" >&2
    return 1
    fi
    detokenizer.perl -l ${tgt_lang} -q < ${ckpt_prefix}.${type}.hyp > ${ckpt_prefix}.${type}.hyp.detok
    detokenizer.perl -l ${tgt_lang} -q < ${ckpt_prefix}.${type}.ref > ${ckpt_prefix}.${type}.ref.detok
    echo "Detokenized BLEU:"
    sacrebleu ${ckpt_prefix}.${type}.ref.detok -i ${ckpt_prefix}.${type}.hyp.detok


done

exit 0
