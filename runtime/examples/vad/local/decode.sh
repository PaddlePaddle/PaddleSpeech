#!/bin/bash
set -e

conf=conf
data=data
exp=exp

. utils/parse_options.sh

mkdir -p $exp
ckpt_dir=$data/silero_vad
model=$ckpt_dir/silero_vad.onnx
test_wav=$data/silero_vad_sample.wav
conf_file=$conf/vad.ini


vad_nnet_main $model $test_wav
echo "vad_nnet_main done!"

vad_interface_main $conf_file $test_wav
echo "vad_interface_main done!"


