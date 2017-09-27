#! /usr/bin/env bash

. ../../utils/utility.sh

URL='http://cloud.dlnel.org/filepub/?uuid=8e3cf742-2ff3-41ce-a49d-f6158cc06a23'
MD5=2ef08f8b608a7c555592161fc14d81a6
TARGET=./librispeech_model.tar.gz


echo "Download LibriSpeech model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download LibriSpeech model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
