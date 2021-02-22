#!/bin/bash

source path.sh

# prepare data
bash ./local/data.sh

# test pretrain model
bash ./local/test_golden.sh

# test pretain model
bash ./local/infer_golden.sh

# train model
bash ./local/train.sh

# test model
bash ./local/test.sh

# infer model
bash ./local/infer.sh
