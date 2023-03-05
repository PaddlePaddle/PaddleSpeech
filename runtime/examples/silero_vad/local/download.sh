#!/bin/bash

mkdir -p data
cd data

wget -c https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad.tgz

test -e silero_vad || tar zxvf silero_vad.tgz

wget -c https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad_sample.wav
