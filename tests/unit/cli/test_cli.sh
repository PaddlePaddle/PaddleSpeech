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
paddlespeech tts --voc mb_melgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --voc style_melgan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --voc hifigan_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
paddlespeech tts --am fastspeech2_aishell3 --voc hifigan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."
paddlespeech tts --am fastspeech2_ljspeech --voc hifigan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."
paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "Life was like a box of chocolates, you never know what you're gonna get." --lang en --spk_id 0
paddlespeech tts --am fastspeech2_vctk --voc hifigan_vctk --input "Life was like a box of chocolates, you never know what you're gonna get." --lang en --spk_id 0
paddlespeech tts --am tacotron2_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am tacotron2_csmsc --voc wavernn_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
paddlespeech tts --am tacotron2_ljspeech --voc pwgan_ljspeech --lang en --input "Life was like a box of chocolates, you never know what you're gonna get."


# Speech Translation (only support linux)
paddlespeech st --input ./en.wav


# batch process
echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts

# shell pipeline
paddlespeech asr --input ./zh.wav | paddlespeech text --task punc

# stats
paddlespeech stats --task asr
paddlespeech stats --task tts
paddlespeech stats --task cls

# Speaker Verification 
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
paddlespeech vector --task spk --input 85236145389.wav

echo -e "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job
paddlespeech vector --task spk --input vec.job

echo -e "demo3 85236145389.wav \n demo4 85236145389.wav" | paddlespeech vector --task spk
rm 85236145389.wav 
rm vec.job


