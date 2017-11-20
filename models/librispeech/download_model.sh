#! /usr/bin/env bash

. ../../utils/utility.sh

URL='http://cloud.dlnel.org/filepub/?uuid=117cde63-cd59-4948-8b80-df782555f7d6'
MD5=1f72d0c5591f453362f0caa09dd57618
TARGET=./librispeech_model.tar.gz


echo "Download LibriSpeech model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download LibriSpeech model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
