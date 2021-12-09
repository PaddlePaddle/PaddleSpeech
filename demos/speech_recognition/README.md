# ASR(Automatic Speech Recognition)

## Introduction
ASR, or Automatic Speech Recognition, refers to the problem of getting a program to automatically transcribe spoken language (speech-to-text). 

This demo is an implementation to recognize text from a specific audio file. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
```bash
pip install paddlespeech
```

### 2. Prepare Input File
Input of this demo should be a WAV file(`.wav`), and the sample rate must be same as the model's.

Here are sample files for this demo that can be downloaded:
```bash
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  paddlespeech asr --input ~/zh.wav
  ```
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
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.

  Output:
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

  Output:
  ```bash
  ASR Result:
  我认为跑步最重要的就是给我带来了身体健康
  ```


### 4.Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python api:

| Model | Language | Sample Rate
| :--- | :---: | :---: |
| conformer_wenetspeech| zh| 16000
| transformer_aishell| zh| 16000
