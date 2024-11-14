(简体中文|[English](./README.md))

# 语音自监督学习
## 介绍
语音自监督学习，指的是在大规模无标记的语音数据集上的训练方法。用这种方法训练出来的模型可以产生很好的声学表征。并且可以通过在有标签的数据集上进行微调，应用于其他下游的语音任务。

这个 demo 是通过语音自监督模型将一个特定的音频文件识别成文本或产生声学表征，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。

## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
这个 demo 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 demo 的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```
### 3. 使用方法
- 命令行 (推荐使用)
  ```bash

  # 识别文本
  paddlespeech ssl --task asr --lang en --input ./en.wav

  # 产生声学表征
  paddlespeech ssl --task vector --lang en --input ./en.wav
  ```
  
  使用方法：
  ```bash
  paddlespeech asr --help
  ```
  参数：
  - `input`(必须输入)：用于识别的音频文件。
  - `model`：ASR 任务的模型，默认值：`wav2vec2`, 可选项：[wav2vec2, hubert, wavlm]。
  - `task`：输出类别，默认值：`asr`。
  - `lang`：模型语言，默认值：`en`。
  - `sample_rate`：音频采样率，默认值：`16000`。
  - `config`：ASR 任务的参数文件，若不设置则使用预训练模型中的默认配置，默认值：`None`。
  - `ckpt_path`：模型参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `yes`；不需要设置额外的参数，一旦设置了该参数，说明你默认同意程序的所有请求，其中包括自动转换输入音频的采样率。默认值：`False`。
  - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。
  - `verbose`: 如果使用，显示 logger 信息。


- Python API
  ```python
  import paddle
  from paddlespeech.cli.ssl import SSLExecutor

  ssl_executor = SSLExecutor()

  # 识别文本
  text = ssl_executor(
      model='wav2vec2',
      task='asr',
      lang='en',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./en.wav',
      device=paddle.get_device())
  print('ASR Result: \n{}'.format(text))

  # 得到声学表征
  feature = ssl_executor(
      model='wav2vec2',
      task='vector',
      lang='en',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./en.wav',
      device=paddle.get_device())
  print('Representation: \n{}'.format(feature))
  ```


  输出：
  ```bash
  ASR Result:
  i knocked at the door on the ancient side of the building
  
  Representation:
  Tensor(shape=[1, 164, 1024], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[[ 0.02351918, -0.12980647,  0.17868176, ...,  0.10118122,
          -0.04614586,  0.17853957],
         [ 0.02361383, -0.12978461,  0.17870593, ...,  0.10103855,
          -0.04638699,  0.17855372],
         [ 0.02345137, -0.12982975,  0.17883906, ...,  0.10104341,
          -0.04643029,  0.17856732],
         ...,
         [ 0.02313030, -0.12918393,  0.17845058, ...,  0.10073373,
          -0.04701405,  0.17862988],
         [ 0.02176583, -0.12929161,  0.17797582, ...,  0.10097728,
          -0.04687393,  0.17864393],
         [ 0.05269200,  0.01297141, -0.23336855, ..., -0.11257174,
          -0.17227529,  0.20338398]]])
  ```
