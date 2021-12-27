(简体中文|[English](./README.md))
# 语音翻译

## 介绍
语音翻译是将会话口语短语翻译成另一语言的过程。

该 Demo 是从特定音频文件中识别文本并将其翻译为目标语言的实现。它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。

## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
这个 Demo 的输入是 WAV(`.wav`) 语音文件

这里给出一些样例文件供 Demo 使用：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

### 3. 使用方法 (暂不支持Windows)
- 命令行(推荐使用)
  ```bash
  paddlespeech st --input ./en.wav
  ```
  使用方法：
  ```bash
  paddlespeech st --help
  ```
  参数：
  - `input`(必须输入)：用于翻译的音频。
  - `model`： 语音翻译的模型类型. 默认：`fat_st_ted`。
  - `src_lang`： 源语言. 默认：`en`。
  - `tgt_lang`： 目标语言. 默认：`zh`。
  - `sample_rate`：输入音频的采样率. 默认：`16000`。
  - `config`：语音翻译任务的配置文件. 如果没有默认使用预训练模型的配置文件. 默认：`None`。
  - `ckpt_path`：模型文件. 如果没有默认使用预训练模型. 默认：`None`。
  - `device`：选择执行的设备. 默认： 当前环境 paddlepaddle 的默认设备。

  输出：
  ```bash
  [2021-12-09 11:13:03,178] [    INFO] [utils.py] [L225] - ST Result: ['我 在 这栋 建筑 的 古老 门上 敲门 。']
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import STExecutor
  
  st_executor = STExecutor()
  text = st_executor(
      model='fat_st_ted',
      src_lang='en',
      tgt_lang='zh',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./en.wav',
      device=paddle.get_device())
  print('ST Result: \n{}'.format(text))
  ```

  输出：
  ```bash
  ST Result:
  ['我 在 这栋 建筑 的 古老 门上 敲门 。'] 
  ```

### 4. 预训练模型

以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

| 模型 | 源语言 | 目标语言
| :--- | :---: | :---: |
| fat_st_ted| en| zh
