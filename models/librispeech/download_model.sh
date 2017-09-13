#! /usr/bin/bash

source ../../utils/utility.sh

# TODO: add urls
URL='to-be-added'
MD5=5b4af224b26c1dc4dd972b7d32f2f52a
TARGET=./librispeech_model.tar.gz


echo "Download LibriSpeech model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download LibriSpeech model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
