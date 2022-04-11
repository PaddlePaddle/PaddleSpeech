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
data=$PWD/data
aishell_wav_scp=aishell_test.scp
if [ ! -d $data/test ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
    unzip -d $data aishell_test.zip
    realpath $data/test/*/*.wav > $data/wavlist
    awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
    paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
fi

model_dir=$PWD/aishell_ds2_online_model
if [ ! -d $model_dir ]; then
    mkdir -p $model_dir 
    wget -P $model_dir -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    tar xzfv $model_dir/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz -C $model_dir
fi

# 3. make feature
aishell_online_model=$model_dir/exp/deepspeech2_online/checkpoints
lm_model_dir=../paddle_asr_model
label_file=./aishell_result
wer=./aishell_wer

nj=40
export GLOG_logtostderr=1

./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

data=$PWD/data
# 3. gen linear feat
cmvn=$PWD/cmvn.ark
cmvn_json2binary_main --json_file=$model_dir/data/mean_std.json --cmvn_write_path=$cmvn

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/feat_log \
linear_spectrogram_without_db_norm_main \
    --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
    --feature_wspecifier=ark,scp:$data/split${nj}/JOB/feat.ark,$data/split${nj}/JOB/feat.scp \
    --cmvn_file=$cmvn \
    --streaming_chunk=0.36

text=$data/test/text

# 4. recognizer
utils/run.pl JOB=1:$nj $data/split${nj}/JOB/log \
  offline_decoder_sliding_chunk_main \
    --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
    --model_path=$aishell_online_model/avg_1.jit.pdmodel \
    --param_path=$aishell_online_model/avg_1.jit.pdiparams \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    --dict_file=$lm_model_dir/vocab.txt \
    --lm_path=$lm_model_dir/avg_1.jit.klm \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result

cat $data/split${nj}/*/result > $label_file

local/compute-wer.py --char=1 --v=1 $label_file $text > $wer
tail $wer
