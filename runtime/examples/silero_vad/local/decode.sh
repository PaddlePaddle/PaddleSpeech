#!/bin/bash
set -e

data=data
exp=exp

. utils/parse_options.sh

mkdir -p $exp
ckpt_dir=$data/silero_vad
model=$ckpt_dir/silero_vad.onnx
test_wav=$data/silero_vad_sample.wav


silero_vad_main $model $test_wav

echo "silero vad done!"