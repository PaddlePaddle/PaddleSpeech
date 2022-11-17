(简体中文|[English](./README.md))

# Whisper模型
## 介绍
Whisper是一种通用的语音识别模型。它是在多种音频的大数据集上训练的，也是一个多任务模型，可以执行多语言语音识别以及语音翻译和语言识别。

Whisper模型由OpenAI Whisper训练 https://github.com/openai/whisper

## 使用方法
### 1. 安装
 请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

 你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
 这个 demo 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

 可以下载此 demo 的示例音频：
 ```bash
 wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
 ```

### 3. 使用方法
 - 命令行 (推荐使用)
   ```bash

   # 识别文本
   paddlespeech whisper --task transcribe --input ./zh.wav

   # 将语音翻译成英语
   paddlespeech whisper --task translate --input ./zh.wav
   ```
  使用方法：
   ```bash
   paddlespeech whisper --help
   ```
   参数：
   - `input`(必须输入)：用于识别的音频文件。
   - `model`：ASR 任务的模型，默认值：`whisper-large`。
   - `task`：输出类别，默认值：`transcribe`。
   - `lang`：模型语言，默认值：`None`，强制设定识别出的语言，默认为模型自行判定。
   - `sample_rate`：音频采样率，默认值：`16000`，目前Whisper暂不支持其他采样率。
   - `config`：ASR 任务的参数文件，若不设置则使用预训练模型中的默认配置，默认值：`None`。
   - `ckpt_path`：模型参数文件，若不设置则下载解码模型使用，默认值：`None`。
   - `yes`；不需要设置额外的参数，一旦设置了该参数，说明你默认同意程序的所有请求，其中包括自动转换输入音频的采样率。默认值：`False`。
   - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。
   - `verbose`: 如果使用，显示 logger 信息。


- Python API
   ```python
   import paddle
   from paddlespeech.cli.whisper import WhisperExecutor

   whisper_executor = WhisperExecutor()

   # 识别文本
   text = whisper_executor(
       model='whisper-large',
       task='transcribe',
       sample_rate=16000,
       config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
       ckpt_path=None,
       audio_file='./zh.wav',
       device=paddle.get_device())
   print('ASR Result: \n{}'.format(text))

    # 将语音翻译成英语
   feature = ssl_executor(
       model='whisper-large',
       task='translate',
       sample_rate=16000,
       config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
       ckpt_path=None,
       audio_file='./zh.wav',
       device=paddle.get_device())
   print('Representation: \n{}'.format(feature))
   ```


   输出：
   ```bash
   Transcribe Result:
   Detected language: Chinese
   [00:00.000 --> 00:05.000] 我认为跑步最重要的就是给我带来了身体健康
   {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 5.0, 'text': '我认为跑步最重要的就是给我带来了身体健康', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50614], 'temperature': 0.0, 'avg_logprob': -0.23577967557040128, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.028302080929279327}], 'language': 'zh'}

   Translate Result:
   Detected language: Chinese
   [00:00.000 --> 00:05.000]  I think the most important thing about running is that it brings me good health.
   {'text': ' I think the most important thing about running is that it brings me good health.', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 5.0, 'text': ' I think the most important thing about running is that it brings me good health.', 'tokens': [50364, 286, 519, 264, 881, 1021, 551, 466, 2614, 307, 300, 309, 5607, 385, 665, 1585, 13, 50614], 'temperature': 0.0, 'avg_logprob': -0.47945233395225123, 'compression_ratio': 1.095890410958904, 'no_speech_prob': 0.028302080929279327}], 'language': 'zh'}