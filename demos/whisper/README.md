([简体中文](./README_cn.md)|English)

## Introduction
Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

Whisper model trained by OpenAI whisper https://github.com/openai/whisper

## Usage
 ### 1. Installation
 see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

 You can choose one way from easy, meduim and hard to install paddlespeech.

 ### 2. Prepare Input File
 The input of this demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

 Here are sample files for this demo that can be downloaded:
 ```bash
 wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
 ```

 ### 3. Usage
 - Command Line(Recommended)
   ```bash
   # to recognize text 
   paddlespeech whisper --task transcribe --input ./zh.wav

   # to change model English-Only base size model
   paddlespeech whisper --lang en --size base --task transcribe  --input ./en.wav

   # to recognize text and translate to English
   paddlespeech whisper --task translate --input ./zh.wav
   
   ```

   Usage:
   ```bash
   paddlespeech whisper --help
   ```
   Arguments:
   - `input`(required): Audio file to recognize.
   - `model`: Model type of asr task. Default: `whisper-large`.
   - `task`: Output type. Default: `transcribe`.
   - `lang`: Model language. Default: ``. Use `en` to choice English-only model. Now [medium,base,small,tiny] size can support English-only.
   - `size`: Model size for decode. Defalut: `large`. Now can support [large,medium,base,small,tiny].
   - `language`: Set decode language. Default: `None`. Forcibly set the recognized language, which is determined by the model itself by default. 
   - `sample_rate`: Sample rate of the model. Default: `16000`. Other sampling rates are not supported now.
   - `config`: Config of asr task. Use pretrained model when it is None. Default: `None`.
   - `ckpt_path`: Model checkpoint. Use pretrained model when it is None. Default: `None`.
   - `yes`: No additional parameters required. Once set this parameter, it means accepting the request of the program by default, which includes transforming the audio sample rate. Default: `False`.
   - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.
   - `verbose`: Show the log information.


 - Python API
   ```python
   import paddle
   from paddlespeech.cli.whisper import WhisperExecutor

   whisper_executor = WhisperExecutor()

   # to recognize text 
   text = whisper_executor(
       model='whisper',
       task='transcribe',
       sample_rate=16000,
       config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
       ckpt_path=None,
       audio_file='./zh.wav',
       device=paddle.get_device())
   print('ASR Result: \n{}'.format(text))

   # to recognize text and translate to English
   feature = whisper_executor(
       model='whisper',
       task='translate',
       sample_rate=16000,
       config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
       ckpt_path=None,
       audio_file='./zh.wav',
       device=paddle.get_device())
   print('Representation: \n{}'.format(feature))
   ```

   Output:
   ```bash
   Transcribe Result:
   Detected language: Chinese
   [00:00.000 --> 00:05.000] 我认为跑步最重要的就是给我带来了身体健康
   {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 5.0, 'text': '我认为跑步最重要的就是给我带来了身体健康', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50614], 'temperature': 0.0, 'avg_logprob': -0.23577967557040128, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.028302080929279327}], 'language': 'zh'}

   Translate Result:
   Detected language: Chinese
   [00:00.000 --> 00:05.000]  I think the most important thing about running is that it brings me good health.
   {'text': ' I think the most important thing about running is that it brings me good health.', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 5.0, 'text': ' I think the most important thing about running is that it brings me good health.', 'tokens': [50364, 286, 519, 264, 881, 1021, 551, 466, 2614, 307, 300, 309, 5607, 385, 665, 1585, 13, 50614], 'temperature': 0.0, 'avg_logprob': -0.47945233395225123, 'compression_ratio': 1.095890410958904, 'no_speech_prob': 0.028302080929279327}], 'language': 'zh'}
