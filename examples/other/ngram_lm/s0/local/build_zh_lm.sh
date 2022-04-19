#!/bin/bash
set -e

stage=0
stop_stage=100

order=5
mem=80%
prune=0
a=22
q=8
b=8

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "$0 token_type exp/text exp/text.arpa"
    echo $@
    exit 1
fi

# char or word
type=$1
text=$2
arpa=$3

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    # text tn & wordseg preprocess
    echo "process text."
    python3 ${MAIN_ROOT}/utils/zh_tn.py --token_type ${type} ${text} ${text}.${type}.tn
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    # train ngram lm
    echo "build lm."
    bash ${MAIN_ROOT}/utils/ngram_train.sh --order ${order} --mem ${mem} --prune "${prune}" ${text}.${type}.tn ${arpa}
fi