#!/bin/bash

mkdir -p data
wav_en=data/en.wav
wav_zh=data/zh.wav
test -e ${wav_en}  || wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav -P data
test -e ${wav_zh}  || wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav -P data

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
if [ ${ngpu} == 0 ];then
    device=cpu
else
    device=gpu
fi

echo "using ${device}..."

python3 -u hub_infer.py \
--device ${device} \
--wav_en ${wav_en} \
--wav_zh ${wav_zh}

exit 0
