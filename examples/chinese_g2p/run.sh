#!/usr/bin/env bash
source path.sh

stage=-1
stop_stage=100

exp_dir="exp"
data_dir="data"
source ${MAIN_ROOT}/utils/parse_options.sh || exit -1
mkdir -p ${exp_dir}
bash local/prepare_dataset.sh --exp-dir ${exp_dir} --data-dir ${data_dir}

# convert transcription in chinese into pinyin with pypinyin or jieba+pypinyin
filename="000001-010000.txt"
echo "Processing transcriptions..."
python3 local/extract_pinyin.py ${exp_dir}/${filename} ${exp_dir}/"pypinyin_result.txt"
python3 local/extract_pinyin.py --use-jieba ${exp_dir}/${filename} ${exp_dir}/"pypinyin_with_jieba_result.txt"

echo "done"
exit 0
