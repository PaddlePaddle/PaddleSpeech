#!/bin/bash
set -e

source path.sh

# prepare data
bash ./local/data.sh

# train model
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./local/train.sh

# test model
CUDA_VISIBLE_DEVICES=0  bash ./local/test.sh

# infer model
CUDA_VISIBLE_DEVICES=0  bash ./local/infer.sh ckpt/checkpoints/step-3284

# export model
bash ./local/export.sh ckpt/checkpoints/step-3284 jit.model