#!/usr/bin/env bash

. ${MAIN_ROOT}/utils/utility.sh

DOWNLOAD_DIR=$(dirname $0)/../data
mkdir -p ${DOWNLOAD_DIR}

# you may need to pass the authentification to download the data via a browser
URL=https://online-of-baklong.oss-cn-huhehaote.aliyuncs.com/story_resource/BZNSYP.rar

MD5="c4350563bf7dc298f7dd364b2607be83"
TARGET=${DOWNLOAD_DIR}/BZNSYP.rar

echo "Download Baker TTS dataset..."
download ${URL} ${MD5} ${TARGET}
if [ $? -ne 0 ]; then
    echo "Fail to downlaod Baker TTS dataset!"
    exit
fi

exit 0
