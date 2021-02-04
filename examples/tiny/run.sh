#!/bin/bash

source path.sh

# prepare data
bash ./local/run_data.sh

# test pretrain model
bash ./local/run_test_golden.sh

# test pretain model
bash ./local/run_infer_golden.sh

# train model
bash ./local/run_train.sh

# test model
bash ./local/run_test.sh

# infer model
bash ./local/run_infer.sh

# tune model
bash ./local/run_tune.sh
