#! /usr/bin/env bash

source ../../utils/utility.sh

URL='http://cloud.dlnel.org/filepub/?uuid=6c83b9d8-3255-4adf-9726-0fe0be3d0274'
MD5=28521a58552885a81cf92a1e9b133a71
TARGET=./aishell_model.tar.gz


echo "Download Aishell model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download Aishell model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
