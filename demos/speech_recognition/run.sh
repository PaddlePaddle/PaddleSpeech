#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav

# asr
paddlespeech asr --input ./zh.wav


# asr + punc
paddlespeech asr --input ./zh.wav | paddlespeech text --task punc
