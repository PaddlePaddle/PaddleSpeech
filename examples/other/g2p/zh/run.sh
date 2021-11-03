#!/usr/bin/env bash

source path.sh

stage=-1
stop_stage=100

exp_dir=exp
data=data

source ${MAIN_ROOT}/utils/parse_options.sh || exit -1

mkdir -p ${exp_dir}

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ];then
    mkdir -p ${data}
    test -e ${data}/BZNSYP.rar || wget -c https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar -P ${data}
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    echo "stage 0: Extracting Prosody Labeling"
    bash local/prepare_dataset.sh --exp-dir ${exp_dir} --data-dir ${data}
fi

# convert transcription in chinese into pinyin with pypinyin or jieba+pypinyin
filename="000001-010000.txt"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: Processing transcriptions..."
    python3 local/extract_pinyin_label.py ${exp_dir}/${filename} ${exp_dir}/ref.pinyin

    python3 local/convert_transcription.py ${exp_dir}/${filename} ${exp_dir}/trans.pinyin
    python3 local/convert_transcription.py --use-jieba ${exp_dir}/${filename} ${exp_dir}/trans.jieba.pinyin
fi

echo "done"
exit 0
