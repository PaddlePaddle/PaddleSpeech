#!/bin/bash

set -ex

ANDROID_NDK=/workspace/zhanghui/android-sdk/android-ndk-r25c

# Setting up Android toolchanin
ANDROID_ABI=arm64-v8a  # 'arm64-v8a', 'armeabi-v7a'
ANDROID_PLATFORM="android-21"  # API >= 21
ANDROID_STL=c++_shared  # 'c++_shared', 'c++_static'
ANDROID_TOOLCHAIN=clang  # 'clang' only
TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake

# Create build directory
BUILD_ROOT=build/Android
BUILD_DIR=${BUILD_ROOT}/${ANDROID_ABI}-api-21
FASTDEPLOY_INSTALL_DIR="/workspace/zhanghui/paddle/FastDeploy/build/Android/arm64-v8a-api-21/install"

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# CMake configuration with Android toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DANDROID_ABI=${ANDROID_ABI} \
      -DANDROID_NDK=${ANDROID_NDK} \
      -DANDROID_PLATFORM=${ANDROID_PLATFORM} \
      -DANDROID_STL=${ANDROID_STL} \
      -DANDROID_TOOLCHAIN=${ANDROID_TOOLCHAIN} \
      -DBUILD_SHARED_LIBS=OFF \
      -DWITH_ASR=OFF \
      -DWITH_CLS=OFF \
      -DWITH_VAD=ON \
      -DFASTDEPLOY_INSTALL_DIR=${FASTDEPLOY_INSTALL_DIR} \
      -DCMAKE_FIND_DEBUG_MODE=OFF \
      -Wno-dev ../../..

# Build FastDeploy Android C++ SDK
make
