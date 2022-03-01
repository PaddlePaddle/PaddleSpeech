(简体中文|[English](./README.md))

# 语音合成

## 介绍
语音合成是一种自然语言建模过程，其将文本转换为语音以进行音频演示。

这个 demo 是一个从给定文本生成音频的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。

## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 easy，medium，hard 三中方式中选择一种方式安装。

### 2. 准备输入

这个 demo 的输入是通过参数传递的特定语言的文本。
### 3. 使用方法
- 命令行 (推荐使用)
    - 中文
    
       默认的声学模型是 `Fastspeech2`，默认的声码器是 `Parallel WaveGAN`.
        ```bash
        paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！"
        ```
    - 批处理
        ```bash
        echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts
        ```
    - 中文，使用 `SpeedySpeech` 作为声学模型
        ```bash
        paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
        ```
    - 中文， 多说话人
    
        你可以改变 `spk_id` 。
        ```bash
        paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
        ```
    
     - 英文
        ```bash
        paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "hello world"
        ```
    - 英文，多说话人
    
        你可以改变 `spk_id` 。
        ```bash
        paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "hello, boys" --lang en --spk_id 0
        ```
  使用方法：
  
  ```bash
  paddlespeech tts --help
  ```
  参数：
  - `input`(必须输入)：用于合成音频的文本。
  - `am`：TTS 任务的声学模型， 默认值：`fastspeech2_csmsc`。
  - `am_config`：声学模型的配置文件，若不设置则使用默认配置，默认值：`None`。
  - `am_ckpt`：声学模型的参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `am_stat`：训练声学模型时用于正则化 mel 频谱图的均值标准差文件，默认值：`None`。
  - `phones_dict`：音素词表文件， 默认值：`None`。
  - `tones_dict`：声调词表文件， 默认值：`None`。
  - `speaker_dict`：说话人词表文件， 默认值：`None`。
  - `spk_id`：说话人 id， 默认值： `0`。
  - `voc`：TTS 任务的声码器， 默认值： `pwgan_csmsc`。
  - `voc_config`：声码器的配置文件，若不设置则使用默认配置，默认值：`None`。
  - `voc_ckpt`：声码器的参数文件，若不设置则下载预训练模型使用，默认值：`None`。
  - `voc_stat`：训练声码器时用于正则化 mel 频谱图的均值标准差文件，默认值：`None`。
  - `lang`：TTS 任务的语言， 默认值：`zh`。
  - `device`：执行预测的设备， 默认值：当前系统下 paddlepaddle 的默认 device。
  - `output`：输出音频的路径， 默认值：`output.wav`。

  输出：
  ```bash
  [2021-12-09 20:49:58,955] [    INFO] [log.py] [L57] - Wave file has been generated: output.wav
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import TTSExecutor

  tts_executor = TTSExecutor()
  wav_file = tts_executor(
      text='今天的天气不错啊',
      output='output.wav',
      am='fastspeech2_csmsc',
      am_config=None,
      am_ckpt=None,
      am_stat=None,
      spk_id=0,
      phones_dict=None,
      tones_dict=None,
      speaker_dict=None,
      voc='pwgan_csmsc',
      voc_config=None,
      voc_ckpt=None,
      voc_stat=None,
      lang='zh',
      device=paddle.get_device())
  print('Wave file has been generated: {}'.format(wav_file))
  ```

  输出：
  ```bash
  Wave file has been generated: output.wav
  ```

### 4. 预训练模型
以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

- 声学模型
  | 模型 | 语言
  | :--- | :---: |
  | speedyspeech_csmsc| zh
  | fastspeech2_csmsc| zh
  | fastspeech2_aishell3| zh
  | fastspeech2_ljspeech| en
  | fastspeech2_vctk| en

- 声码器
  | 模型 | 语言
  | :--- | :---: |
  | pwgan_csmsc| zh
  | pwgan_aishell3| zh
  | pwgan_ljspeech| en
  | pwgan_vctk| en
  | mb_melgan_csmsc| zh
