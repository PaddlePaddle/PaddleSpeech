#!/usr/bin/env bash
set -xe

# the build script had verified in the paddlepaddle docker image.
# please follow the instruction below to install PaddlePaddle image.
# https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html 
cmake -B build -DWITH_ASR=OFF -DWITH_CLS=OFF
cmake --build build -j
