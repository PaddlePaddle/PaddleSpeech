#!/usr/bin/env bash

source path.sh

stage=-1
stop_stage=100

exp_dir=exp
data_dir=data
filename="sentences.txt"

source ${MAIN_ROOT}/utils/parse_options.sh || exit -1

mkdir -p ${exp_dir}


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: Processing "
    python3 local/test_normalization.py  ${data_dir}/${filename} ${exp_dir}/normalized.txt
    if [ -f "${exp_dir}/normalized.txt" ]; then
	echo "Normalized text save at ${exp_dir}/normalized.txt"
    fi
    # TODO(chenfeiyu): compute edit distance against ground-truth
fi

echo "done"
exit 0
