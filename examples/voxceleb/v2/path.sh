#!/bin/bash
export MAIN_ROOT=`realpath ${PWD}/../../../`

export PATH=${MAIN_ROOT}:${MAIN_ROOT}/utils:${PATH}

export LC_ALL=C

export PYTHONDONTWRITEBYTECODE=1
# Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/
export LD_LIBRARY_PATH=/mnt/xiongxinlei/opt/anaconda3/envs/paddlespeech/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/xiongxinlei/opt/nccl/lib/:${LD_LIBRARY_PATH}


export BIN_DIR=${MAIN_ROOT}/paddlespeech/vector/bin/