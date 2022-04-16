# This contains the locations of binarys build required for running the examples.

SPEECHX_ROOT=$PWD/../../../
MAIN_ROOT=$SPEECHX_ROOT/../
SPEECHX_EXAMPLES=$SPEECHX_ROOT/build/examples

SPEECHX_TOOLS=$SPEECHX_ROOT/tools
TOOLS_BIN=$SPEECHX_TOOLS/valgrind/install/bin

[ -d $SPEECHX_EXAMPLES ] || { echo "Error: 'build/examples' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

export PATH=$PATH:$TOOLS_BIN

# srilm
export LIBLBFGS=${MAIN_ROOT}/tools/liblbfgs-1.10
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${LIBLBFGS}/lib/.libs
export SRILM=${MAIN_ROOT}/tools/srilm
export PATH=${PATH}:${SRILM}/bin:${SRILM}/bin/i686-m64