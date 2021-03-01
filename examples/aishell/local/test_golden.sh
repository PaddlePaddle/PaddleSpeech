#! /usr/bin/env bash

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

# evaluate model
CUDA_VISIBLE_DEVICES=0 \
python3 -u ${BIN_DIR}/test.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--checkpoint_path data/pretrain/params.pdparams  \
--opts data.mean_std_filepath data/pretrain/mean_std.npz  \
--opts data.vocab_filepath data/pretrain/vocab.txt

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
