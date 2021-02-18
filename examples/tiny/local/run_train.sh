#! /usr/bin/env bash

# train model
# if you wish to resume from an exists model, uncomment --init_from_pretrained_model
export FLAGS_sync_nccl_allreduce=0

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python3 -u ${MAIN_ROOT}/train.py \
#--num_iter_print=1 \
#--save_epoch=1 \
#--num_samples=64 \
#--test_off=False \
#--is_local=True \
#--output_model_dir="./checkpoints/" \
#--shuffle_method="batch_shuffle_clipped" \

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -u ${MAIN_ROOT}/train.py \
--device 'gpu' \
--nproc 4 \
--config conf/deepspeech2.yaml \
--output ckpt

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
