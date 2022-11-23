([简体中文](./README_cn.md)|English)

# Speech Service

## Introduction

This document introduces a client for streaming asr service: microphone


## Usage
### 1. Install
Refer [Install](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

 **paddlepaddle 2.4rc** 或以上版本。
It is recommended to use **paddlepaddle 2.4rc** or above.
You can choose one way from meduim and hard to install paddlespeech.


### 2. Prepare config File


The input of  ASR client demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for thisASR client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 2. Streaming ASR Client Usage

- microphone
   ```
   python microphone_client.py

   ```
