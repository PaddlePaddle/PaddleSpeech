#!/bin/bash

# audio download
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav

# to recognize text 
paddlespeech whisper --task transcribe --input ./zh.wav

# to recognize text and translate to English
paddlespeech whisper --task translate --input ./zh.wav

# to change model English-Only model
paddlespeech whisper --lang en --size base --task transcribe  --input ./en.wav