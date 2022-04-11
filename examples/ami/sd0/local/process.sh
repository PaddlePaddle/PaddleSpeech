#!/bin/bash

stage=0
set=L

. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
set -o pipefail

data_folder=$1
manual_annot_folder=$2
save_folder=$3
pretrained_model_dir=$4
conf_path=$5
device=$6

ref_rttm_dir=${save_folder}/ref_rttms
meta_data_dir=${save_folder}/metadata

if [ ${stage} -le 0 ]; then
    echo "AMI Data preparation"
    python local/ami_prepare.py  --data_folder ${data_folder} \
            --manual_annot_folder ${manual_annot_folder} \
            --save_folder ${save_folder} --ref_rttm_dir ${ref_rttm_dir} \
            --meta_data_dir ${meta_data_dir} 
    
    if [ $? -ne 0 ]; then
        echo "Prepare AMI failed. Please check log message."
        exit 1
    fi
    echo "AMI data preparation done."           
fi

if [ ${stage} -le 1 ]; then
    # extra embddings for dev and eval dataset
    for name in dev eval; do
        python local/compute_embdding.py --config ${conf_path} \
                --data-dir ${save_folder} \
                --device ${device} \
                --dataset ${name} \
                --load-checkpoint ${pretrained_model_dir}
    done
fi

if [ ${stage} -le 2 ]; then
    # tune hyperparams on dev set
    # perform final diarization on 'dev' and 'eval' with best hyperparams
    python local/experiment.py --config ${conf_path} \
            --data-dir ${save_folder}
fi
