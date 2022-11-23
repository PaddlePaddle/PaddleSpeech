(简体中文|[English](./PPTTS.md))

# PP-TTS

- [1. 简介](#1)
- [2. 特性](#2)
- [3. Benchmark](#3)
- [4. 效果展示](#4)
- [5. 使用教程](#5)
    - [5.1 模型训练与推理优化](#51)
    - [5.2 语音合成特色应用](#52)
    - [5.3 语音合成服务搭建](#53)

<a name="1"></a>
## 1. 简介

PP-TTS 是 PaddleSpeech 自研的流式语音合成系统。在实现[前沿算法](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#text-to-speech-models)的基础上，使用了更快的推理引擎，实现了流式语音合成技术，使其满足商业语音交互场景的需求。

#### PP-TTS
语音合成基本流程如下图所示：
<center><img src=https://ai-studio-static-online.cdn.bcebos.com/ea69ae1faff84940a59c7079d16b3a8db2741d2c423846f68822f4a7f28726e9 width="600" ></center>

PP-TTS 默认提供基于 FastSpeech2 声学模型和 HiFiGAN 声码器的中文流式语音合成系统：

- 文本前端：采用基于规则的中文文本前端系统，对文本正则、多音字、变调等中文文本场景进行了优化。
- 声学模型：对 FastSpeech2 模型的 Decoder 进行改进，使其可以流式合成
- 声码器：支持对 GAN Vocoder 的流式合成
- 推理引擎：使用 ONNXRuntime 推理引擎优化模型推理性能，使得语音合成系统在低压 CPU 上也能达到 RTF<1，满足流式合成的要求

<a name="2"></a>
## 2. 特性
- 开源领先的中文语音合成系统
- 使用 ONNXRuntime 推理引擎优化模型推理性能
- 唯一开源的流式语音合成系统
- 易拆卸性：可以很方便地更换不同语种上的不同声学模型和声码器、使用不同的推理引擎（Paddle 动态图、PaddleInference 和 ONNXRuntime 等）、使用不同的网络服务（HTTP、Websocket）

<a name="3"></a>
## 3. Benchmark
PaddleSpeech TTS 模型之间的性能对比，请查看 [TTS-Benchmark](https://github.com/PaddlePaddle/PaddleSpeech/wiki/TTS-Benchmark)。

<a name="4"></a>
## 4. 效果展示 
请参考：[Streaming TTS Demo Video](https://paddlespeech.readthedocs.io/en/latest/streaming_tts_demo_video.html)

<a name="5"></a>
## 5. 使用教程

<a name="51"></a>
### 5.1 模型训练与推理优化

Default FastSpeech2：[tts3/run.sh](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/tts3/run.sh)

流式 FastSpeech2：[tts3/run_cnndecoder.sh](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/tts3/run_cnndecoder.sh)

HiFiGAN：[voc5/run.sh](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/voc5/run.sh)

<a name="52"></a>
### 5.2 语音合成特色应用
一键式实现语音合成：[text_to_speech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/text_to_speech)

个性化语音合成 - 基于 FastSpeech2 模型的个性化语音合成：[style_fs2](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/style_fs2)

会说话的故事书 - 基于 OCR 和语音合成的会说话的故事书：[story_talker](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/story_talker)

元宇宙 - 基于语音合成的 2D 增强现实：[metaverse](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/metaverse)

<a name="53"></a>
### 5.3 语音合成服务搭建

一键式搭建非流式语音合成服务：[speech_server](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/speech_server)

一键式搭建流式语音合成服务：[streaming_tts_server](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/streaming_tts_server)


更多教程，包括模型设计、模型训练、推理部署等，请参考 AIStudio 教程：[PP-TTS：流式语音合成原理及服务部署
](https://aistudio.baidu.com/aistudio/projectdetail/3885352)
