#!/bin/bash

wget -c https://paddlespeech.cdn.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.cdn.bcebos.com/PaddleAudio/dog.wav
paddlespeech cls --input ./cat.wav --topk 10
