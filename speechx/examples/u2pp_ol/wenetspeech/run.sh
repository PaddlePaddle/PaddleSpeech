#!/bin/bash
set -e

. path.sh

nj=40
stage=-1
stop_stage=100

. utils/parse_options.sh

# input
data=data
exp=exp
mkdir -p $exp $data
aishell_wav_scp=aishell_test.scp

# 1. compile
if [ ! -d ${SPEECHX_BUILD} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi


ckpt_dir=$data/model

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    #  download u2pp model
    if [ ! -f $ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model.tar.gz ]; then
        mkdir -p $ckpt_dir
        pushd $ckpt_dir

        wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/static/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model.tar.gz
        tar xzfv asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model.tar.gz

        popd
    fi

    # download u2pp quant model
    if [ ! -f $ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_quant_1.3.0.model.tar.gz ]; then
        mkdir -p $ckpt_dir
        pushd $ckpt_dir

        wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/static/asr1_chunk_conformer_u2pp_wenetspeech_static_quant_1.3.0.model.tar.gz
        tar xzfv asr1_chunk_conformer_u2pp_wenetspeech_static_quant_1.3.0.model.tar.gz

        popd
    fi

    # test wav scp
    if [ ! -f data/wav.scp ]; then
        mkdir -p $data
        pushd $data
        wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
        echo "utt1 " $PWD/zh.wav > wav.scp
        popd 
    fi

    # aishell wav scp
    if [ ! -d $data/test ]; then
        pushd $data
        wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
        unzip  aishell_test.zip
        popd

        realpath $data/test/*/*.wav > $data/wavlist
        awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
        paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # process cmvn and compute fbank feat
    ./local/feat.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # decode with fbank feat input
    ./local/decode.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # decode with wav input
    ./loca/recognizer.sh
fi
