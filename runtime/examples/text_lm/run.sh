#!/bin/bash
set -eo pipefail

. path.sh

stage=0
stop_stage=100
has_key=true
token_type=word

. utils/parse_options.sh || exit -1;

text=data/text

if [ ! -f $text ]; then
    echo "$0: Not find $1";
    exit -1;
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    echo "text tn & wordseg preprocess"
    rm -rf ${text}.tn
    python3 utils/zh_tn.py --has_key $has_key --token_type $token_type ${text} ${text}.tn
fi