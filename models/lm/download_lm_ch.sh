#! /usr/bin/env bash

. ../../utils/utility.sh

URL=http://cloud.dlnel.org/filepub/?uuid=d21861e4-4ed6-45bb-ad8e-ae417a43195e
MD5="29e02312deb2e59b3c8686c7966d4fe3"
TARGET=./zh_giga.no_cna_cmn.prune01244.klm


echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
fi


exit 0
