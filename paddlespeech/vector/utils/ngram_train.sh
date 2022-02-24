#!/bin/bash

set -e

order=5
mem=80%
prune=0
a=22
q=8
b=8

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "$0 exp/text exp/text.arpa"
    exit 1
fi

text=${1}
arpa=${2}
lmbin=${2}.klm.bin

# https://kheafield.com/code/kenlm/estimation/
echo "build arpa lm."
lmplz -o ${order} -S ${mem} --prune ${prune} < ${text} > ${arpa} || { echo "train kenlm error!"; exit -1; }

# https://kheafield.com/code/kenlm/
echo "build binary lm."
build_binary -a ${a} -q ${q} -b ${b} trie ${arpa} ${lmbin} || { echo "build kenlm binary error!"; exit -1; }