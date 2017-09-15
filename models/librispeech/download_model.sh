#! /usr/bin/bash

source ../../utils/utility.sh

# TODO: add urls
URL='http://cloud.dlnel.org/filepub/?uuid=17404caf-cf19-492f-9707-1fad07c19aae'
MD5=ea5024a457a91179472f6dfee60e053d
TARGET=./librispeech_model.tar.gz


echo "Download LibriSpeech model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download LibriSpeech model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
