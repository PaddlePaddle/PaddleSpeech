#!/bin/bash

. ./path.sh || exit 1;
set -e

stage=1
stop_stage=50

#TARGET_DIR=${MAIN_ROOT}/dataset/ami
TARGET_DIR=/home/dataset/AMI
data_folder=${TARGET_DIR}/amicorpus #e.g., /path/to/amicorpus/
manual_annot_folder=${TARGET_DIR}/ami_public_manual_1.6.2 #e.g., /path/to/ami_public_manual_1.6.2/

save_folder=./save
pretraind_model_dir=${save_folder}/model

conf_path=conf/ecapa_tdnn.yaml


. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Prepare data and model
    # Download AMI corpus, You need around 10GB of free space to get whole data
    # The signals are too large to package in this way,
    # so you need to use the chooser to indicate which ones you wish to download
    echo "Please follow https://groups.inf.ed.ac.uk/ami/download/ to download the data."
    echo "Annotations: AMI manual annotations v1.6.2 "
    echo "Signals: "
    echo "1) Select one or more AMI meetings: the IDs please follow ./ami_split.py"
    echo "2) Select media streams: Just select Headset mix"
    # Download the pretrained Model from HuggingFace or other pretrained model
    echo "Please download the pretrained ECAPA-TDNN Model and put the pretrainde model in given path: "${pretraind_model_dir}
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Tune hyperparams on dev set and perform final diarization on dev and eval with best hyperparams.
    bash ./local/process.sh ${data_folder} ${manual_annot_folder} ${save_folder} ${pretraind_model_dir} ${conf_path} || exit 1
fi

