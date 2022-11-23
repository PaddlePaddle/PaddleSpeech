([简体中文](./README_cn.md)|English)
# Speech SSL (Self-Supervised Learning)

## Introduction
Speech SSL, or Self-Supervised Learning, refers to a training method on the large-scale unlabeled speech dataset. The model trained in this way can produce a good acoustic representation, and can be applied to other downstream speech tasks by fine-tuning on labeled datasets.

This demo is an implementation to recognize text or produce the acoustic representation from a specific audio file by speech ssl models. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  # to recognize text 
  paddlespeech ssl --task asr --lang en --input ./en.wav

  # to get acoustic representation
  paddlespeech ssl --task vector --lang en --input ./en.wav
  ```

  Usage:
  ```bash
  paddlespeech ssl --help
  ```
  Arguments:
  - `input`(required): Audio file to recognize.
  - `model`: Model type of asr task. Default: `wav2vec2ASR_librispeech`.
  - `task`: Output type. Default: `asr`.
  - `lang`: Model language. Default: `en`.
  - `sample_rate`: Sample rate of the model. Default: `16000`.
  - `config`: Config of asr task. Use pretrained model when it is None. Default: `None`.
  - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `yes`: No additional parameters required. Once set this parameter, it means accepting the request of the program by default, which includes transforming the audio sample rate. Default: `False`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.
  - `verbose`: Show the log information.


- Python API
  ```python
  import paddle
  from paddlespeech.cli.ssl import SSLExecutor

  ssl_executor = SSLExecutor()

  # to recognize text 
  text = ssl_executor(
      model='wav2vec2ASR_librispeech',
      task='asr',
      lang='en',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./en.wav',
      device=paddle.get_device())
  print('ASR Result: \n{}'.format(text))

  # to get acoustic representation
  feature = ssl_executor(
      model='wav2vec2',
      task='vector',
      lang='en',
      sample_rate=16000,
      config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
      ckpt_path=None,
      audio_file='./en.wav',
      device=paddle.get_device())
  print('Representation: \n{}'.format(feature))
  ```

  Output:
  ```bash
  ASR Result:
<<<<<<< HEAD
  我认为跑步最重要的就是给我带来了身体健康
=======
  i knocked at the door on the ancient side of the building
>>>>>>> 45426846942f68cf43a23677d8d55f6d4ab93ab1

  Representation:
  Tensor(shape=[1, 164, 1024], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[[ 0.02351918, -0.12980647,  0.17868176, ...,  0.10118122,
          -0.04614586,  0.17853957],
         [ 0.02361383, -0.12978461,  0.17870593, ...,  0.10103855,
          -0.04638699,  0.17855372],
         [ 0.02345137, -0.12982975,  0.17883906, ...,  0.10104341,
          -0.04643029,  0.17856732],
         ...,
         [ 0.02313030, -0.12918393,  0.17845058, ...,  0.10073373,
          -0.04701405,  0.17862988],
         [ 0.02176583, -0.12929161,  0.17797582, ...,  0.10097728,
          -0.04687393,  0.17864393],
         [ 0.05269200,  0.01297141, -0.23336855, ..., -0.11257174,
          -0.17227529,  0.20338398]]])
  ```
