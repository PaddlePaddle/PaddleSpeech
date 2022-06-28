(简体中文|[English](./README.md))

# 关键词识别
## 介绍
关键词识别是一项用于识别一段语音内是否包含特定的关键词。

这个 demo 是一个从给定音频文件识别特定关键词的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。
## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入
这个 demo 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 demo 的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/kws/hey_snips.wav https://paddlespeech.bj.bcebos.com/kws/non-keyword.wav
```
### 3. 使用方法
- 命令行 (推荐使用)
  ```bash
  paddlespeech kws --input ./hey_snips.wav
  paddlespeech kws --input ./non-keyword.wav
  ```
  
  使用方法：
  ```bash
  paddlespeech kws --help
  ```
  参数：
  - `input`(必须输入)：用于识别关键词的音频文件。
  - `threshold`：用于判别是包含关键词的得分阈值，默认值：`0.8`。
  - `model`：KWS 任务的模型，默认值：`mdtc_heysnips`。
  - `config`：KWS 任务的参数文件，若不设置则使用预训练模型中的默认配置，默认值：`None`。
  - `ckpt_path`：模型参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `device`：执行预测的设备，默认值：当前系统下 paddlepaddle 的默认 device。
  - `verbose`: 如果使用，显示 logger 信息。

  输出：
  ```bash
  # 输入为 ./hey_snips.wav
  Score: 1.000, Threshold: 0.8, Is keyword: True
  # 输入为 ./non-keyword.wav
  Score: 0.000, Threshold: 0.8, Is keyword: False
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli.kws import KWSExecutor

  kws_executor = KWSExecutor()
  result = kws_executor(
      audio_file='./hey_snips.wav',
      threshold=0.8,
      model='mdtc_heysnips',
      config=None,
      ckpt_path=None,
      device=paddle.get_device())
  print('KWS Result: \n{}'.format(result))
  ```

  输出：
  ```bash
  KWS Result:
  Score: 1.000, Threshold: 0.8, Is keyword: True
  ```

### 4.预训练模型
以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

| 模型 | 语言 | 采样率
| :--- | :---: | :---: |
| mdtc_heysnips | en | 16k
