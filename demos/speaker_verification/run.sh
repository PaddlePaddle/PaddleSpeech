#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/123456789.wav

# vector
paddlespeech vector --task spk --input ./85236145389.wav

paddlespeech vector --task score --input "./85236145389.wav ./123456789.wav"
