export MAIN_ROOT=${PWD}/../../../

export PATH=${MAIN_ROOT}:${MAIN_ROOT}/utils:${PATH}
export LC_ALL=C

# Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/

# model exp
MODEL=u2
export BIN_DIR=${MAIN_ROOT}/deepspeech/exps/${MODEL}/bin


export LIBLBFGS=/workspace/zhanghui/asr/wenet-210713/tools/liblbfgs-1.10
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${LIBLBFGS}/lib/.libs
export SRILM=/workspace/zhanghui/asr/wenet-210713/tools/srilm
export PATH=${PATH}:${SRILM}/bin:${SRILM}/bin/i686-m64

# Kaldi
export KALDI_ROOT=/workspace/zhanghui/asr/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
