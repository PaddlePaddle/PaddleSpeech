#!/bin/bash

# audio download
wget -c https://paddlespeech.cdn.bcebos.com/PaddleAudio/en.wav

# to recognize text 
paddlespeech ssl --task asr --lang en --input ./en.wav

# to get acoustic representation
paddlespeech ssl --task vector --lang en --input ./en.wav
