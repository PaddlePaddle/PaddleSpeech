# PaddleSpeech TTS CPP Frontend

A TTS frontend that implements text-to-phoneme conversion.

Currently it only supports Chinese, any English word will crash the demo.

## Install Build Tools

```
# Ubuntu
sudo apt install build-essential cmake pkg-config

# CentOS
sudo yum groupinstall "Development Tools"
sudo yum install cmake
```

If your cmake version is too old, you can go here to download a precompiled new version: https://cmake.org/download/

## Build

```
# Build with all CPU cores
./build.sh

# Build with 1 core
./build.sh -j1
```

Dependent libraries will be automatically downloaded to the `third-party/build` folder.

If the download speed is too slow, you can open [third-party/CMakeLists.txt](third-party/CMakeLists.txt) and modify `GIT_REPOSITORY` URLs.

## Run

```
./run_front_demo.sh
./run_front_demo.sh --help
./run_front_demo.sh --sentence "这是语音合成服务的文本前端，用于将文本转换为音素序号数组。"
./run_front_demo.sh --front_conf ./front_demo/front.conf --sentence "你还需要一个语音合成后端才能将其转换为实际的声音。"
```

## Clean

```
./clean.sh
```

The folders `build` and `third-party/build` will be deleted.
