#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav

# asr
paddlespeech vector --task spk --input ./85236145389.wav