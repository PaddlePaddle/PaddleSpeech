#!/bin/bash

wget -c https://paddlespeech.cdn.bcebos.com/kws/hey_snips.wav https://paddlespeech.cdn.bcebos.com/kws/non-keyword.wav

# kws
paddlespeech kws --input ./hey_snips.wav
paddlespeech kws --input non-keyword.wav
