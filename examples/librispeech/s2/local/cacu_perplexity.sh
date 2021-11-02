#!/bin/bash

set -e

stage=-1
stop_stage=100

expdir=exp
datadir=data

ngpu=0

# lm params
rnnlm_config_path=conf/lm/transformer.yaml
lmexpdir=exp/lm/transformer
lang_model=transformerLM.pdparams

#data path
test_set=${datadir}/test_clean/text
test_set_lower=${datadir}/test_clean/text_lower
train_set=train_960

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
bpeprefix=${datadir}/lang_char/${train_set}_${bpemode}${nbpe}
bpemodel=${bpeprefix}.model

vocabfile=${bpeprefix}_units.txt
vocabfile_lower=${bpeprefix}_units_lower.txt

output_dir=${expdir}/lm/transformer/perplexity

mkdir -p ${output_dir}

# Transform the data upper case to lower
if [ -f ${vocabfile} ]; then
    tr A-Z a-z < ${vocabfile} > ${vocabfile_lower}
fi

if [ -f ${test_set} ]; then
    tr A-Z a-z < ${test_set} > ${test_set_lower}
fi

python ${LM_BIN_DIR}/cacu_perplexity.py \
    --rnnlm ${lmexpdir}/${lang_model} \
    --rnnlm-conf ${rnnlm_config_path} \
    --vocab_path ${vocabfile_lower} \
    --bpeprefix ${bpeprefix} \
    --text_path ${test_set_lower} \
    --output_dir ${output_dir} \
    --ngpu ${ngpu}

