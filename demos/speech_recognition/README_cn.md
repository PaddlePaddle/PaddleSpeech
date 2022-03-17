(简体中文|[English](./README.md))

# 语音识别
## 介绍
语音识别是一项用计算机程序自动转录语音的技术。

这个 demo 是一个从给定音频文件识别文本的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。
## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
这个 demo 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 demo 的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```
### 3. 使用方法
- 命令行 (推荐使用)
  ```bash
  # 中文
  paddlespeech asr --input ./zh.wav
  # 英文
  paddlespeech asr --model transformer_librispeech --lang en --input ./en.wav
  # 中文 + 标点恢复
  paddlespeech asr --input ./zh.wav | paddlespeech text --task punc
  ```
  (如果显示 `paddlespeech-ctcdecoders` 这个 python 包没有找到的 Error，没有关系，这个包是非必须的。)
  
  使用方法：
  ```bash
  paddlespeech asr --help
  ```
  参数：
  - `input`(必须输入)：用于识别的音频文件。
  - `model`：ASR 任务的模型，默认值：`conformer_wenetspeech`。
  - `lang`：模型语言，默认值：`zh`。
  - `sample_rate`：音频采样率，默认值：`16000`。
  - `config`：ASR 任务的参数文件，若不设置则使用预训练模型中的默认配置，默认值：`None`。
  - `ckpt_path`：模型参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `yes`；不需要设置额外的参数，一旦设置了该参数，说明你默认同意程序的所有请求，其中包括自动转换输入音频的采样率。默认值：`False`。
  - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。

  输出：
  ```bash
  # 中文
  [2021-12-08 13:12:34,063] [    INFO] [utils.py] [L225] - ASR Result: 我认为跑步最重要的就是给我带来了身体健康
  # 英文
  [2022-01-12 11:51:10,815] [    INFO] - ASR Result: i knocked at the door on the ancient side of the building
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import ASRExecutor

  asr_executor = ASRExecutor()
  text = asr_executor(
      model='conformer_wenetspeech',
      lang='zh',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./zh.wav',
      force_yes=False,
      device=paddle.get_device())
  print('ASR Result: \n{}'.format(text))
  ```

  输出：
  ```bash
  ASR Result:
  我认为跑步最重要的就是给我带来了身体健康
  ```

### 4.预训练模型
以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

| 模型 | 语言 | 采样率
| :--- | :---: | :---: |
| conformer_wenetspeech | zh | 16k
| transformer_librispeech | en | 16k
| deepspeech2offline_aishell| zh| 16k
| deepspeech2online_aishell | zh | 16k
| deepspeech2offline_librispeech | en | 16k
