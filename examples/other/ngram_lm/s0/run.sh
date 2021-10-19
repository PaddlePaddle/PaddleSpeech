#!/bin/bash
set -e
source path.sh

stage=0
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh || exit -1

python3 -c 'import kenlm;' || { echo "kenlm package not install!"; exit -1; }

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    # case 1, test kenlm
    # download language model
    bash local/download_lm_zh.sh
    if [ $? -ne 0 ]; then
       exit 1
    fi

    # test kenlm `score` and `full_score`
    python local/kenlm_score_test.py data/lm/zh_giga.no_cna_cmn.prune01244.klm
fi

mkdir -p exp
cp data/text_correct.txt exp/text

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    # case 2, chinese chararctor ngram lm build
    # output: xxx.arpa xxx.kenlm.bin
    input=exp/text
    token_type=char
    lang=zh
    order=5
    prune="0 1 2 4 4"
    a=22
    q=8
    b=8
    output=${input}_${lang}_${token_type}_o${order}_p${prune// /_}_a${a}_q${q}_b${b}.arpa
    echo "build ${token_type} lm."
    bash local/build_zh_lm.sh --order ${order} --prune "${prune}" --a ${a} --q ${a} --b ${b} ${token_type} ${input} ${output}
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
    # case 2, chinese chararctor ngram lm build
    # output: xxx.arpa xxx.kenlm.bin
    input=exp/text
    token_type=word
    lang=zh
    order=3
    prune="0 0 0"
    a=22
    q=8
    b=8
    output=${input}_${lang}_${token_type}_o${order}_p${prune// /_}_a${a}_q${q}_b${b}.arpa
    echo "build ${token_type} lm."
    bash local/build_zh_lm.sh --order ${order} --prune "${prune}" --a ${a} --q ${a} --b ${b} ${token_type} ${input} ${output}
fi
