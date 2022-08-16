#!/bin/bash

wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/123456789.wav

# sid extract
paddlespeech_client vector --server_ip 127.0.0.1 --port 8090 --task spk --input ./85236145389.wav

# sid score
paddlespeech_client vector --server_ip 127.0.0.1 --port 8090 --task score --enroll ./85236145389.wav --test ./123456789.wav
