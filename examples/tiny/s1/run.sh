#!/bin/bash
set -e

source path.sh

# prepare data
bash ./local/data.sh

# train model
bash ./local/train.sh

# test model
bash ./local/test.sh

# infer model
bash ./local/infer.sh
