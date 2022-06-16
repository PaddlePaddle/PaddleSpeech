# This contains the locations of binarys build required for running the examples.

MAIN_ROOT=`realpath $PWD/../../../../`
SPEECHX_ROOT=$PWD/../../../
SPEECHX_BUILD=$SPEECHX_ROOT/build/speechx

SPEECHX_TOOLS=$SPEECHX_ROOT/tools
TOOLS_BIN=$SPEECHX_TOOLS/valgrind/install/bin

[ -d $SPEECHX_BUILD ] || { echo "Error: 'build/speechx' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

# openfst bin & kaldi bin
KALDI_DIR=$SPEECHX_ROOT/build/speechx/kaldi/
OPENFST_DIR=$SPEECHX_ROOT/fc_patch/openfst-build/src

# srilm
export LIBLBFGS=${MAIN_ROOT}/tools/liblbfgs-1.10
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${LIBLBFGS}/lib/.libs
export SRILM=${MAIN_ROOT}/tools/srilm

SPEECHX_BIN=$SPEECHX_BUILD/decoder:$SPEECHX_BUILD/frontend/audio
export PATH=$PATH:$SPEECHX_BIN:$TOOLS_BIN:${SRILM}/bin:${SRILM}/bin/i686-m64:$KALDI_DIR/lmbin:$KALDI_DIR/fstbin:$OPENFST_DIR/bin
