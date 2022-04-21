#!/bin/bash
set +x
set -e

. path.sh

# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# input
mkdir -p data
data=$PWD/data
ckpt_dir=$data/model
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/
vocb_dir=$ckpt_dir/data/lang_char/

lm=$data/zh_giga.no_cna_cmn.prune01244.klm

# output
exp_dir=./exp
mkdir -p $exp_dir

# 2. download model
if [[ ! -f data/model/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz ]]; then
    mkdir -p data/model
    pushd data/model
    wget -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    tar xzfv asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    popd
fi

# produce wav scp
if [ ! -f data/wav.scp ]; then
    pushd data
    wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
    echo "utt1 " $PWD/zh.wav > wav.scp
    popd 
fi

# download lm
if [ ! -f $lm ]; then
    pushd data
    wget -c https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
    popd
fi

feat_wspecifier=$exp_dir/feats.ark
cmvn=$exp_dir/cmvn.ark

export GLOG_logtostderr=1

# dump json cmvn to kaldi
cmvn-json2kaldi \
    --json_file  $ckpt_dir/data/mean_std.json \
    --cmvn_write_path $cmvn \
    --binary=false
echo "convert json cmvn to kaldi ark."


# generate linear feature as streaming
linear-spectrogram-wo-db-norm-ol \
    --wav_rspecifier=scp:$data/wav.scp \
    --feature_wspecifier=ark,t:$feat_wspecifier \
    --cmvn_file=$cmvn
echo "compute linear spectrogram feature."

# run ctc beam search decoder as streaming
ctc-prefix-beam-search-decoder-ol \
    --result_wspecifier=ark,t:$exp_dir/result.txt \
    --feature_rspecifier=ark:$feat_wspecifier \
    --model_path=$model_dir/avg_1.jit.pdmodel \
    --param_path=$model_dir/avg_1.jit.pdiparams \
    --dict_file=$vocb_dir/vocab.txt \
    --lm_path=$lm