#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav
paddlespeech cls --input ./cat.wav --topk 10
