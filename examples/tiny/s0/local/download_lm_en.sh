#!/bin/bash

. ${MAIN_ROOT}/utils/utility.sh

DIR=data/lm
mkdir -p ${DIR}

URL=https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm
MD5="099a601759d467cd0a8523ff939819c5"
TARGET=${DIR}/common_crawl_00.prune01111.trie.klm

echo "Start downloading the language model. The language model is large, please wait for a moment ..."
download $URL $MD5 $TARGET > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
else
    echo "Download the language model sucessfully"
fi


exit 0
