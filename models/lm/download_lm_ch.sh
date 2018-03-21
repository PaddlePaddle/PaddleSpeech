#! /usr/bin/env bash

. ../../utils/utility.sh

URL=http://cloud.dlnel.org/filepub/?uuid=5cd1688e-78d9-4b9e-9c2f-6f104bd5b518
MD5="29e02312deb2e59b3c8686c7966d4fe3"
TARGET=./zh_giga.no_cna_cmn.prune01244.klm


echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
fi


exit 0
