#!/bin/bash
export MAIN_ROOT=`realpath ${PWD}/../../../`

export PATH=${MAIN_ROOT}:${MAIN_ROOT}/utils:${PATH}
export LC_ALL=C

export PYTHONDONTWRITEBYTECODE=1
# Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}

MODEL=panns
export BIN_DIR=${MAIN_ROOT}/paddlespeech/cls/exps/${MODEL}