#!/bin/bash

# wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav

# asr
export CUDA_VISIBLE_DEVICES=0
paddlespeech asr --input audio/119994.wav -v


# asr + punc
# paddlespeech asr --input ./zh.wav | paddlespeech text --task punc