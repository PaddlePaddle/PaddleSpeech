#!/usr/bin/env bash
set -xe

# the build script had verified in the paddlepaddle docker image.
# please follow the instruction below to install PaddlePaddle image.
# https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html 
#cmake -B build -DBUILD_SHARED_LIBS=OFF -DWITH_ASR=OFF -DWITH_CLS=OFF -DWITH_VAD=ON -DFASTDEPLOY_INSTALL_DIR=/workspace/zhanghui/paddle/FastDeploy/build/Android/arm64-v8a-api-21/install
cmake -B build -DBUILD_SHARED_LIBS=OFF -DWITH_ASR=OFF -DWITH_CLS=OFF -DWITH_VAD=ON -DFASTDEPLOY_INSTALL_DIR=/workspace/zhanghui/paddle/FastDeploy/build/Linux/x86_64/install
cmake --build build -j
