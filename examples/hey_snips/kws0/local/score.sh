#!/bin/bash

python3 ${BIN_DIR}/score.py --cfg_path=$1

python3 ${BIN_DIR}/compute_det.py --cfg_path=$1
