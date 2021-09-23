#!/bin/bash

. ${MAIN_ROOT}/utils/utility.sh

DIR=data/lm
mkdir -p ${DIR}

URL=https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm
MD5="099a601759d467cd0a8523ff939819c5"
TARGET=${DIR}/common_crawl_00.prune01111.trie.klm

echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
fi


exit 0
