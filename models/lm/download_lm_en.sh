#! /usr/bin/env bash

. ../../utils/utility.sh

URL=http://paddlepaddle.bj.bcebos.com/model_zoo/speech/common_crawl_00.prune01111.trie.klm
MD5="099a601759d467cd0a8523ff939819c5"
TARGET=./common_crawl_00.prune01111.trie.klm


echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
fi


exit 0
