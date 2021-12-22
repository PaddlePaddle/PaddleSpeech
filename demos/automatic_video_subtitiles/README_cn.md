(简体中文|[English](./README.md))
# 视频字幕生成
## 介绍
视频字幕生成可以使用语音识别系统从特定视频生成字幕。

这个 demo 是一个为视频自动生成字幕的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。
## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
获取包含特定语言语音的视频文件：
```bash
wget -c https://paddlespeech.bj.bcebos.com/demos/asr_demos/subtitle_demo1.mp4
```
从视频文件中提取单通道的 16kHz 采样率的 `.wav` 文件：
```bash
ffmpeg -i subtitle_demo1.mp4 -ac 1 -ar 16000 -vn input.wav
```
### 3. 使用方法
- Python API
  ```python
  import paddle
  from paddlespeech.cli import ASRExecutor, TextExecutor

  asr_executor = ASRExecutor()
  text_executor = TextExecutor()

  text = asr_executor(
      audio_file='input.wav',
      device=paddle.get_device())
  result = text_executor(
      text=text,
      task='punc',
      model='ernie_linear_p3_wudao',
      device=paddle.get_device())
  print('Text Result: \n{}'.format(result))
  ```
  输出:
  ```bash
  Text Result:
  当我说我可以把三十年的经验变成一个准确的算法，他们说不可能。当我说我们十个人就能实现对十九个城市变电站七乘二十四小时的实时监管，他们说不可能。
  ```
