export MAIN_ROOT=`realpath ${PWD}/../../../`
export LOCAL_DEEPSPEECH2=`realpath ${PWD}/../`

export PATH=${MAIN_ROOT}:${MAIN_ROOT}/utils:${PATH}
export LC_ALL=C

export PYTHONDONTWRITEBYTECODE=1
# Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}
export PYTHONPATH=${LOCAL_DEEPSPEECH2}:${PYTHONPATH}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/

MODEL=deepspeech2
export BIN_DIR=${LOCAL_DEEPSPEECH2}/src_deepspeech2x/bin
