#!/bin/bash

stage=0
. ./path.sh
export CUDA_VISIBLE_DEVICES=1
if [ $stage -le 0 ]; then
#    /home/users/xiongxinlei/.conda/envs/paddlespeech/bin/python3\
    /home/users/public/miniconda3/envs/paddlespeech/bin/python\
         train_speaker_embeddings.py \
         hparams/train_ecapa_tdnn.yaml \
         --device "gpu:0" \
        --debug \
        --debug_batches 10
        #  --skip_prep false \
        #  --number_of_epochs 10
    #/home/users/xiongxinlei/.conda/envs/paddlespeech/bin/python3 train_speaker_embeddings.py 
fi
