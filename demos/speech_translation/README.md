([简体中文](./README_cn.md)|English)
# Speech Translation
## Introduction
Speech translation is the process by which conversational spoken phrases are instantly translated and spoken aloud in a second language.

This demo is an implementation to recognize text from a specific audio file and translate it to the target language. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.


### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`).

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

### 3. Usage (not support for Windows now)
- Command Line(Recommended)
  ```bash
  paddlespeech st --input ./en.wav
  ```
  Usage:
  ```bash
  paddlespeech st --help
  ```
  Arguments:
  - `input`(required): Audio file to recognize and translate.
  - `model`: Model type of st task. Default: `fat_st_ted`.
  - `src_lang`: Source language. Default: `en`.
  - `tgt_lang`: Target language. Default: `zh`.
  - `sample_rate`: Sample rate of the model. Default: `16000`.
  - `config`: Config of st task. Use pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.

  Output:
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

  Output:
  ```bash
  ST Result:
  ['我 在 这栋 建筑 的 古老 门上 敲门 。'] 
  ```

### 4.Pretrained Models
Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

| Model | Source Language | Target Language
| :--- | :---: | :---: |
| fat_st_ted| en| zh
