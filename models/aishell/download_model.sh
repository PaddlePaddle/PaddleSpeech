#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/mandarin_models/aishell_model_fluid.tar.gz'
MD5=2bf0cc8b6d5da2a2a787b5cc36a496b5
TARGET=./aishell_model_fluid.tar.gz


echo "Download Aishell model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download Aishell model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
