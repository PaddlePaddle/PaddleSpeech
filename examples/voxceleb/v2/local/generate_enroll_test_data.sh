#!/bin/bash


stage=-1
cmvn=true

. ${MAIN_ROOT}/utils/parse_options.sh || exit -1;
voxceleb1_root=$1
trial=$2
dir=$3
config_conf=$4
if [ ${stage} -le 0 ]; then
    python3 local/generate_enroll_test_data.py \
                                --voxceleb1 ${voxceleb1_root} \
                                --trial ${trial} \
                                --dir ${dir}
fi

if [ ${stage} -le 1 ]; then
    python3 ./local/compute_feature.py --config ${config_conf} \
                --dataset ./data/enroll/data.json --feat ./data/enroll/data.feat


    python3 ./local/compute_feature.py --config ${config_conf} \
                --dataset ./data/test/data.json --feat ./data/test/data.feat

fi


if [ ${stage} -le 2 ]; then
    echo "apply the cmvn"
    python3 ./local/apply_cmvn.py --cmvn data/cmvn.feat \
            --feat ./data/enroll/data.feat --target ./data/enroll/apply_cmvn_data.feat
    
    python3 ./local/apply_cmvn.py --cmvn data/cmvn.feat \
            --feat ./data/test/data.feat --target ./data/test/apply_cmvn_data.feat
fi
