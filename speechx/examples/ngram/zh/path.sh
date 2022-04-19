# This contains the locations of binarys build required for running the examples.

MAIN_ROOT=`realpath $PWD/../../../../`
SPEECHX_ROOT=`realpath $MAIN_ROOT/speechx`

export LC_AL=C

# srilm
export LIBLBFGS=${MAIN_ROOT}/tools/liblbfgs-1.10
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${LIBLBFGS}/lib/.libs
export SRILM=${MAIN_ROOT}/tools/srilm
export PATH=${PATH}:${SRILM}/bin:${SRILM}/bin/i686-m64
