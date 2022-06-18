#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav

# If `127.0.0.1` is not accessible, you need to use the actual service IP address.
paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input ./zh.wav --topk 1
