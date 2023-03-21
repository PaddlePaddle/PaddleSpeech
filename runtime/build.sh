#!/usr/bin/env bash
set -xe

BUILD_ROOT=build/Linux
BUILD_DIR=${BUILD_ROOT}/x86_64

mkdir -p ${BUILD_DIR}

BUILD_TYPE=Release
#BUILD_TYPE=Debug
BUILD_SO=OFF
BUILD_ASR=ON
BUILD_CLS=ON
BUILD_VAD=ON
FASTDEPLOY_INSTALL_DIR=""

# the build script had verified in the paddlepaddle docker image.
# please follow the instruction below to install PaddlePaddle image.
# https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html 
#cmake -B build -DBUILD_SHARED_LIBS=OFF -DWITH_ASR=OFF -DWITH_CLS=OFF -DWITH_VAD=ON -DFASTDEPLOY_INSTALL_DIR=/workspace/zhanghui/paddle/FastDeploy/build/Android/arm64-v8a-api-21/install
cmake -B ${BUILD_DIR} \
       	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
       	-DBUILD_SHARED_LIBS=${BUILD_SO} \
	-DWITH_ASR=${BUILD_ASR} \
	-DWITH_CLS=${BUILD_CLS} \
	-DWITH_VAD=${BUILD_VAD} \
	-DFASTDEPLOY_INSTALL_DIR=${FASTDEPLOY_INSTALL_DIR}

cmake --build ${BUILD_DIR} -j
