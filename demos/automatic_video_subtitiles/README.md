# Automatic Video Subtitiles

## Introduction
Automatic video subtitiles can generate subtitiles from a specific video by using Automatic Speech Recognition (ASR) system. 

This demo is an implementation to automatic video subtitiles from a video file. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
```bash
pip install paddlespeech
```

### 2. Prepare Input
Get a video file with speech of the specific language:
```bash
wget -c https://paddlespeech.bj.bcebos.com/demos/asr_demos/subtitle_demo1.mp4
```

Extract `.wav` with one channel and 16000 sample rate from the video:
```bash
ffmpeg -i subtitle_demo1.mp4 -ac 1 -ar 16000 -vn input.wav
```


### 3. Usage

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
  Output:
  ```bash
  Text Result:
  当我说我可以把三十年的经验变成一个准确的算法，他们说不可能。当我说我们十个人就能实现对十九个城市变电站七乘二十四小时的实时监管，他们说不可能。
  ```

automatic_video_subtitiles
