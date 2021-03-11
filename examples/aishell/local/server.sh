#! /usr/bin/env bash
# TODO: replace the model with a mandarin model

if [[ $# != 1 ]];then
   echo "usage: $1 checkpoint_path"
   exit -1
fi

source path.sh

# download language model
bash local/download_lm_ch.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# download well-trained model
bash local/download_model.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# start demo server
CUDA_VISIBLE_DEVICES=0 \
python3 -u ${BIN_DIR}/deploy/server.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--host_ip="localhost" \
--host_port=8086 \
--speech_save_dir="demo_cache" \
--checkpoint_path ${1} 

if [ $? -ne 0 ]; then
    echo "Failed in starting demo server!"
    exit 1
fi


exit 0
