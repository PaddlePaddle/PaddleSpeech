#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/eng_models/librispeech_v1.8_to_v2.x.tar.gz'
MD5=7b0f582fe2f5a840b840e7ee52246bc5
TARGET=./librispeech_v1.8_to_v2.x.tar.gz


echo "Download LibriSpeech model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download LibriSpeech model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
