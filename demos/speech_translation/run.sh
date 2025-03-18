#!/bin/bash

wget -c https://paddlespeech.cdn.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.cdn.bcebos.com/PaddleAudio/en.wav
paddlespeech st --input ./en.wav
