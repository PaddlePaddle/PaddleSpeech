#!/bin/bash
set +x
set -e

. ./path.sh

# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# 2. download model
if [ ! -e data/model/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz ]; then
    mkdir -p data/model
    pushd data/model
    wget -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    tar xzfv asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    popd
fi

# produce wav scp
if [ ! -f data/wav.scp ]; then
    mkdir -p data
    pushd data
    wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
    echo "utt1 " $PWD/zh.wav > wav.scp
    popd 
fi


# input
data_dir=./data
exp_dir=./exp
model_dir=$data_dir/model/

mkdir -p $exp_dir


# 3. run feat
export GLOG_logtostderr=1

cmvn_json2kaldi_main \
    --json_file=$model_dir/data/mean_std.json \
    --cmvn_write_path=$exp_dir/cmvn.ark \
    --binary=false
echo "convert json cmvn to kaldi ark."


compute_linear_spectrogram_main \
    --wav_rspecifier=scp:$data_dir/wav.scp \
    --feature_wspecifier=ark,t:$exp_dir/feats.ark \
    --cmvn_file=$exp_dir/cmvn.ark
echo "compute linear spectrogram feature."

compute_fbank_main \
    --num_bins=161 \
    --wav_rspecifier=scp:$data_dir/wav.scp \
    --feature_wspecifier=ark,t:$exp_dir/fbank.ark \
    --cmvn_file=$exp_dir/cmvn.ark
echo "compute fbank feature."

