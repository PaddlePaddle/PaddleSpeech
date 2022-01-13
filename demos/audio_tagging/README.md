([简体中文](./README_cn.md)|English)

# Audio Tagging

## Introduction
Audio tagging is the task of labeling an audio clip with one or more labels or tags, including music tagging, acoustic scene classification, audio event classification, etc.

This demo is an implementation to tag an audio file with 527 [AudioSet](https://research.google.com/audioset/) labels. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`).

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  paddlespeech cls --input ./cat.wav --topk 10
  ```
  Usage:
  ```bash
  paddlespeech cls --help
  ```
  Arguments:
  - `input`(required): The audio file to tag.
  - `model`: Model type of tagging task. Default: `panns_cnn14`.
  - `config`: Config of tagging task. Use a pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use a pretrained model when it is None. Default: `None`.
  - `label_file`: Label file of tagging task. Use audio set labels when it is None. Default: `None`.
  - `topk`: Show topk tagging labels of the result. Default: `1`.
  - `device`: Choose the device to execute model inference. Default: default device of paddlepaddle in the current environment.

  Output:
  ```bash
  [2021-12-08 14:49:40,671] [    INFO] [utils.py] [L225] - CLS Result:
  Cat: 0.8991316556930542
  Domestic animals, pets: 0.8806838393211365
  Meow: 0.8784668445587158
  Animal: 0.8776564598083496
  Caterwaul: 0.2232048511505127
  Speech: 0.03101264126598835
  Music: 0.02870696596801281
  Inside, small room: 0.016673989593982697
  Purr: 0.008387474343180656
  Bird: 0.006304860580712557
  ```

- Python API
  ```python
  import paddle
  from paddlespeech.cli import CLSExecutor

  cls_executor = CLSExecutor()
  result = cls_executor(
      model='panns_cnn14',
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      label_file=None,
      ckpt_path=None,
      audio_file='./cat.wav',
      topk=10,
      device=paddle.get_device())
  print('CLS Result: \n{}'.format(result))
  ```
  Output:
  ```bash
  CLS Result:
  Cat: 0.8991316556930542
  Domestic animals, pets: 0.8806838393211365
  Meow: 0.8784668445587158
  Animal: 0.8776564598083496
  Caterwaul: 0.2232048511505127
  Speech: 0.03101264126598835
  Music: 0.02870696596801281
  Inside, small room: 0.016673989593982697
  Purr: 0.008387474343180656
  Bird: 0.006304860580712557
  ```

### 4.Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python API:

| Model | Sample Rate
| :--- | :---: 
| panns_cnn6| 32000
| panns_cnn10| 32000
| panns_cnn14| 32000
