#! /usr/bin/env bash

. ../../utils/utility.sh

URL='http://cloud.dlnel.org/filepub/?uuid=61de63b9-6904-4809-ad95-0cc5104ab973'
MD5=0ee83aa15fba421e5de8fc66c8feb350
TARGET=./aishell_model.tar.gz


echo "Download Aishell model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download Aishell model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
