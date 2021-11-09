#!/bin/bash

. ${MAIN_ROOT}/utils/utility.sh

DIR=data/lm
mkdir -p ${DIR}

URL='https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
MD5="29e02312deb2e59b3c8686c7966d4fe3"
TARGET=${DIR}/zh_giga.no_cna_cmn.prune01244.klm

echo "Start downloading the language model. The language model is large, please wait for a moment ..."
download $URL $MD5 $TARGET > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
else
    echo "Download the language model sucessfully"
fi


exit 0
