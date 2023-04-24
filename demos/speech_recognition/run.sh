#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav

# asr
paddlespeech asr --input ./zh.wav


# asr + punc
paddlespeech asr --input ./zh.wav | paddlespeech text --task punc


# asr help
paddlespeech asr --help


# english asr
paddlespeech asr --lang en --model transformer_librispeech --input ./en.wav

# model stats
paddlespeech stats --task asr


# paddlespeech help
paddlespeech --help
