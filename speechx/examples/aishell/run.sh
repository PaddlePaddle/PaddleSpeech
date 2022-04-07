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


# 2. download model
if [ ! -d ../paddle_asr_model ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/paddle_asr_model.tar.gz
    tar xzfv paddle_asr_model.tar.gz
    mv ./paddle_asr_model ../
    # produce wav scp
    echo "utt1 " $PWD/../paddle_asr_model/BAC009S0764W0290.wav > ../paddle_asr_model/wav.scp
fi

mkdir -p data
if [ ! -d ./test ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
    unzip aishell_test.zip
    realpath ./test/*/*.wav > wavlist
    awk -F '/' '{ print $(NF) }' wavlist | awk -F '.' '{ print $1 }' > utt_id
    paste utt_id wavlist > aishell_test.scp
fi

if [ ! -d aishell_ds2_online_model ]; then
    mkdir -p aishell_ds2_online_model 
    wget -P ./aishell_ds2_online_model -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/aishell_ds2_online_cer8.00_release.tar.gz
    tar xzfv ./aishell_ds2_online_model/aishell_ds2_online_cer8.00_release.tar.gz -C ./aishell_ds2_online_model
fi

# 3. make feature
aishell_wav_scp=./aishell_test.scp
aishell_online_model=./aishell_ds2_online_model/exp/deepspeech2_online/checkpoints
model_dir=../paddle_asr_model
feat_ark=./feats.ark
feat_scp=./aishell_feat.scp
cmvn=./cmvn.ark
label_file=./aishell_result
wer=./aishell_wer

export GLOG_logtostderr=1

# 3. gen linear feat
linear_spectrogram_main \
    --wav_rspecifier=scp:$aishell_wav_scp \
    --feature_wspecifier=ark,scp:$feat_ark,$feat_scp \
    --cmvn_write_path=$cmvn \
    --streaming_chunk=10

nj=10
data=./data
text=./test/text
# recognizer
./local/split_data.sh data aishell_feat.scp $nj

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/log \
  offline_decoder_sliding_chunk_main \
    --feature_rspecifier=scp:$data/split${nj}/JOB/feats.scp \
    --model_path=$aishell_online_model/avg_1.jit.pdmodel \
    --param_path=$aishell_online_model/avg_1.jit.pdiparams \
    --dict_file=$model_dir/vocab.txt \
    --lm_path=$model_dir/avg_1.jit.klm \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result

cat $data/split${nj}/*/result > $label_file

local/compute-wer.py --char=1 --v=1 $label_file $text > $wer
