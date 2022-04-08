([简体中文](./README_cn.md)|English)
# ASR (Automatic Speech Recognition)

## Introduction
ASR, or Automatic Speech Recognition, refers to the problem of getting a program to automatically transcribe spoken language (speech-to-text). 

This demo is an implementation to recognize text from a specific audio file. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  # Chinese
  paddlespeech asr --input ./zh.wav
  # English
  paddlespeech asr --model transformer_librispeech --lang en --input ./en.wav
  # Chinese ASR + Punctuation Restoration
  paddlespeech asr --input ./zh.wav | paddlespeech text --task punc
  ```
  (It doesn't matter if package `paddlespeech-ctcdecoders` is not found, this package is optional.)
  
  Usage:
  ```bash
  paddlespeech asr --help
  ```
  Arguments:
  - `input`(required): Audio file to recognize.
  - `model`: Model type of asr task. Default: `conformer_wenetspeech`.
  - `lang`: Model language. Default: `zh`.
  - `sample_rate`: Sample rate of the model. Default: `16000`.
  - `config`: Config of asr task. Use pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `yes`: No additional parameters required. Once set this parameter, it means accepting the request of the program by default, which includes transforming the audio sample rate. Default: `False`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.

  Output:
  ```bash
  # Chinese
  [2021-12-08 13:12:34,063] [    INFO] [utils.py] [L225] - ASR Result: 我认为跑步最重要的就是给我带来了身体健康
  # English
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

  Output:
  ```bash
  ASR Result:
  我认为跑步最重要的就是给我带来了身体健康
  ```

### 4.Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

| Model | Language | Sample Rate
| :--- | :---: | :---: |
| conformer_wenetspeech| zh| 16k
| transformer_librispeech| en| 16k
| deepspeech2offline_aishell| zh| 16k
| deepspeech2online_aishell | zh | 16k
|deepspeech2offline_librispeech|en| 16k
