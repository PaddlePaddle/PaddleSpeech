(简体中文|[English](./README.md))

# 语音识别
## 介绍
语音识别解决让计算机程序自动转录语音的问题。

这个 demo 是一个从给定音频文件识别文本的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。
## 使用方法
### 1. 安装
```bash
pip install paddlespeech
```
### 2. 准备输入
这个 demo 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 demo 的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```
### 3. 使用方法
- 命令行 (推荐使用)
  ```bash
  paddlespeech asr --input ./zh.wav
  ```
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
  - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。

  输出：
  ```bash
  [2021-12-08 13:12:34,063] [    INFO] [utils.py] [L225] - ASR Result: 我认为跑步最重要的就是给我带来了身体健康
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
| conformer_wenetspeech| zh| 16000
| transformer_aishell| zh| 16000
