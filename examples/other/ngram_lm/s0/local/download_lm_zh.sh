#! /usr/bin/env bash

. ${MAIN_ROOT}/utils/utility.sh

DIR=data/lm
mkdir -p ${DIR}

URL='https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
MD5="29e02312deb2e59b3c8686c7966d4fe3"
TARGET=${DIR}/zh_giga.no_cna_cmn.prune01244.klm


if [ -e $TARGET ];then
    echo "already have lm"
    exit 0;
fi

echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
fi


exit 0
