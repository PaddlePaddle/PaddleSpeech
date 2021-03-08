export MAIN_ROOT=${PWD}

export PATH=${MAIN_ROOT}:${PWD}/tools:${PATH}
export LC_ALL=C

# Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8 
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}

export export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/
