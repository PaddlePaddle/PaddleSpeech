([简体中文](./PPTTS_cn.md)|English)

- [1. Introduction](#1)
- [2. Characteristic](#2)
- [3. Benchmark](#3)
- [4. Demo](#4)
- [5. Tutorials](#5)
    - [5.1 Training and Inference Optimization](#51)
    - [5.2 Characteristic APPs of TTS](#52)
    - [5.3 TTS Server](#53)

<a name="1"></a>
## 1. Introduction

PP-TTS is a streaming speech synthesis system developed by PaddleSpeech. Based on the implementation of [SOTA Algorithms](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#text-to-speech-models), a faster inference engine is used to realize streaming speech synthesis technology to meet the needs of commercial speech interaction scenarios.

#### PP-TTS
Pipline of TTS：
<center><img src=https://ai-studio-static-online.cdn.bcebos.com/ea69ae1faff84940a59c7079d16b3a8db2741d2c423846f68822f4a7f28726e9 width="600" ></center>

PP-TTS provides a Chinese streaming speech synthesis system based on FastSpeech2 and HiFiGAN by default:

- Text Frontend： The rule-based Chinese text frontend system is adopted to optimize Chinese text such as text normalization, polyphony, and tone sandhi.
- Acoustic Model: The decoder of FastSpeech2 is improved so that it can be stream synthesized
- Vocoder: Streaming synthesis of GAN vocoder is supported
- Inference Engine： Using ONNXRuntime to optimize the inference of TTS models, so that the TTS system can also achieve RTF < 1 on low-voltage, meeting the requirements of streaming synthesis

<a name="2"></a>
## 2. Characteristic
- Open source leading Chinese TTS system
- Using ONNXRuntime to optimize the inference of TTS models
- The only open-source streaming TTS system
- Easy disassembly: Developers can easily replace different acoustic models and vocoders in different languages, use different inference engines (Paddle dynamic graph, PaddleInference, ONNXRuntime, etc.), and use different network services (HTTP, WebSocket)

<a name="3"></a>
## 3. Benchmark
PaddleSpeech TTS models' benchmark: [TTS-Benchmark](https://github.com/PaddlePaddle/PaddleSpeech/wiki/TTS-Benchmark)。

<a name="4"></a>
## 4. Demo 
See: [Streaming TTS Demo Video](https://paddlespeech.readthedocs.io/en/latest/streaming_tts_demo_video.html)

<a name="5"></a>
## 5. Tutorials

<a name="51"></a>
### 5.1 Training and Inference Optimization

Default FastSpeech2: [tts3/run.sh](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/tts3/run.sh)

Streaming FastSpeech2: [tts3/run_cnndecoder.sh](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/tts3/run_cnndecoder.sh)

HiFiGAN：[voc5/run.sh](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/csmsc/voc5/run.sh)

<a name="52"></a>
### 5.2 Characteristic APPs of TTS
text_to_speech - convert text into speech: [text_to_speech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/text_to_speech)

style_fs2 - multi style control for FastSpeech2 model: [style_fs2](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/style_fs2)

story talker - book reader based on OCR and TTS: [story_talker](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/story_talker)

metaverse - 2D AR with TTS: [metaverse](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/metaverse)

<a name="53"></a>
### 5.3 TTS Server

Non-streaming TTS Server: [speech_server](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/speech_server)

Streaming TTS Server: [streaming_tts_server](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/demos/streaming_tts_server)


For more tutorials please see: [PP-TTS：流式语音合成原理及服务部署
](https://aistudio.baidu.com/aistudio/projectdetail/3885352)
