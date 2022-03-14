#!/bin/bash

stage=-1
stop_stage=100

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;

dir=$1
conf_path=$2
mkdir -p ${dir}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # data prepare for vox1 and vox2, vox2 must be converted from m4a to wav
    # we should use the local/convert.sh convert m4a to wav
    python3 local/data_prepare.py \
                        --data-dir ${dir} \
                        --config ${conf_path}
fi 