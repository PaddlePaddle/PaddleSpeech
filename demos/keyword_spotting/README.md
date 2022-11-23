([简体中文](./README_cn.md)|English)
# KWS (Keyword Spotting)

## Introduction
KWS(Keyword Spotting) is a technique to recognize keyword from a giving speech audio.

This demo is an implementation to recognize keyword from a specific audio file. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/kws/hey_snips.wav https://paddlespeech.bj.bcebos.com/kws/non-keyword.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  paddlespeech kws --input ./hey_snips.wav
  paddlespeech kws --input ./non-keyword.wav
  ```
  
  Usage:
  ```bash
  paddlespeech kws --help
  ```
  Arguments:
  - `input`(required): Audio file to recognize.
  - `threshold`：Score threshold for kws. Default: `0.8`.
  - `model`: Model type of kws task. Default: `mdtc_heysnips`.
  - `config`: Config of kws task. Use pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.
  - `verbose`: Show the log information.

  Output:
  ```bash
  # Input file: ./hey_snips.wav
  Score: 1.000, Threshold: 0.8, Is keyword: True
  # Input file: ./non-keyword.wav
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

  Output:
  ```bash
  KWS Result:
  Score: 1.000, Threshold: 0.8, Is keyword: True
  ```

### 4.Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

| Model | Language | Sample Rate
| :--- | :---: | :---: |
| mdtc_heysnips | en | 16k
