#!/bin/bash

. ./path.sh || exit 1;
set -e

stage=0

#TARGET_DIR=${MAIN_ROOT}/dataset/ami
TARGET_DIR=/home/dataset/AMI
data_folder=${TARGET_DIR}/amicorpus #e.g., /path/to/amicorpus/
manual_annot_folder=${TARGET_DIR}/ami_public_manual_1.6.2 #e.g., /path/to/ami_public_manual_1.6.2/

save_folder=./save
pretraind_model_dir=${save_folder}/sv0_ecapa_tdnn_voxceleb12_ckpt_0_1_1/model
conf_path=conf/ecapa_tdnn.yaml
device=gpu

. ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

if [ $stage -le 1 ]; then
    # Download the pretrained model
    wget https://paddlespeech.bj.bcebos.com/vector/voxceleb/sv0_ecapa_tdnn_voxceleb12_ckpt_0_1_1.tar.gz
    mkdir -p ${save_folder} && tar -xvf sv0_ecapa_tdnn_voxceleb12_ckpt_0_1_1.tar.gz -C ${save_folder}
    rm -rf sv0_ecapa_tdnn_voxceleb12_ckpt_0_1_1.tar.gz
    echo "download the pretrained ECAPA-TDNN Model to path: "${pretraind_model_dir}
fi

if [ $stage -le 2 ]; then
    # Tune hyperparams on dev set and perform final diarization on dev and eval with best hyperparams.
    echo ${data_folder} ${manual_annot_folder} ${save_folder} ${pretraind_model_dir} ${conf_path}
    bash ./local/process.sh ${data_folder} ${manual_annot_folder} \
        ${save_folder} ${pretraind_model_dir} ${conf_path} ${device} || exit 1
fi

