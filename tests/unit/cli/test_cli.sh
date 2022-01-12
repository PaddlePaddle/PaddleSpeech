#!/bin/bash
set -e
# Audio classification
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav
paddlespeech cls --input ./cat.wav --topk 10

# Punctuation_restoration
paddlespeech text --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭

# Speech_recognition
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
paddlespeech asr --input ./zh.wav
paddlespeech asr --model transformer_librispeech --lang en --input ./en.wav

# Text To Speech
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "hello world"
paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "hello, boys" --lang en --spk_id 0


# Speech Translation (only support linux)
paddlespeech st --input ./en.wav
