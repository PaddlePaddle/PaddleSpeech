#!/bin/bash
set -e

source path.sh

# prepare data
bash ./local/data.sh

# train model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./local/train.sh

# test model
CUDA_VISIBLE_DEVICES=0  bash ./local/test.sh

# infer model
CUDA_VISIBLE_DEVICES=0 bash ./local/infer.sh
