#!/usr/bin/env bash
source path.sh

stage=0
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh || exit -1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    bash local/data_download.sh
    if [ $? -ne 0 ]; then
	exit 1
    fi
fi

EXP_DIR="exp"
mkdir -p ${EXP_DIR}

ARCHIVE="data/BZNSYP.rar"

echo "Extracting Prosody Labeling"
LABEL_FILE='ProsodyLabeling/000001-010000.txt'
FILENAME='000001-010000.txt'
unrar e ${ARCHIVE} ${LABEL_FILE}
mv ${FILENAME} ${EXP_DIR}

if [ ! -f ${EXP_DIR}/${FILENAME} ];then
    echo "File extraction failed!"
    exit
fi

# convert transcription in chinese into pinyin with pypinyin or jieba+pypinyin
python3 local/extract_pinyin.py ${EXP_DIR}/${FILENAME} ${EXP_DIR}/"pypinyin_result.txt"
python3 local/extract_pinyin.py --use-jieba ${EXP_DIR}/${FILENAME} ${EXP_DIR}/"pypinyin_with_jieba_result.txt"

echo "done"
exit 0


    
    
    
    
    
