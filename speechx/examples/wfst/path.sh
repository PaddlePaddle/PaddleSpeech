# This contains the locations of binarys build required for running the examples.

MAIN_ROOT=`realpath $PWD/../../../../`
SPEECHX_ROOT=`realpath $MAIN_ROOT/speechx`

export LC_AL=C

# srilm
export LIBLBFGS=${MAIN_ROOT}/tools/liblbfgs-1.10
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${LIBLBFGS}/lib/.libs
export SRILM=${MAIN_ROOT}/tools/srilm
export PATH=${PATH}:${SRILM}/bin:${SRILM}/bin/i686-m64

# Kaldi
export KALDI_ROOT=${MAIN_ROOT}/tools/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present, can not using Kaldi!"
[ -f $KALDI_ROOT/tools/config/common_path.sh ] && . $KALDI_ROOT/tools/config/common_path.sh
