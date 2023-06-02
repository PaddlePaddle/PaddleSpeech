#!/bin/bash

exp=exp
pretrain=$exp/pretrain

mkdir -p $pretrain

pushd $pretrain

wget -c https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_ckpt_1.2.0.zip &
wget -c https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip &
wait

popd
