
# TTS (Text To Speech)

## Introduction
Text-to-speech (TTS) is a natural language modeling process that requires changing units of text into units of speech for audio presentation. 

This demo is an implementation to generate an audio from the giving text. It can be done by a single command or a few lines in python using `PaddleSpeech`. 

## Usage
### 1. Installation
```bash
pip install paddlespeech
```

### 2. Prepare Input
Input of this demo should be a text of the specific language that can be passed via argument.
### 3. Usage
- Command Line (Recommended)
	- Chinese
	```bash
	paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！"
	```
	The default acoustic model is `Fastspeech2`, and the default vocoder is `Parallel WaveGAN`.
  	- Chinese, use `SpeedySpeech` as acoustic model
	```bash
	paddlespeech tts --am speedyspeech_csmsc --input "你好，欢迎使用百度飞桨深度学习框架！"
	```
	- Chinese, multi speaker
	```bash
	paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --input "你好，欢迎使用百度飞桨深度学习框架！" --spk_id 0
	```
	You can change `spk_id` here.
     - English
	```bash
	paddlespeech tts --am fastspeech2_ljspeech --voc pwgan_ljspeech --lang en --input "hello world"
	```
	- English, multi speaker
	```bash
	paddlespeech tts --am fastspeech2_vctk --voc pwgan_vctk --input "hello, boys" --lang en --spk_id 0
    ```
     You can change `spk_id` here.
	   
 
- Usage:
```bash
  paddlespeech tts --help
  ```
  Arguments:
  - `input`(required): Input text to generate..
  - `am`: Acoustic model type of tts task. Default: `fastspeech2_csmsc`.
  - `am_config`: Config of acoustic model. Use deault config when it is None. Default: `None`.
  - `am_ckpt`: Acoustic model checkpoint. Use pretrained model when it is None. Default: `None`.
  - `am_stat`: Mean and standard deviation used to normalize spectrogram when training acoustic model. Default: `None`.
  - `phones_dict`: Phone vocabulary file. Default: `None`.
  - `tones_dict`: Tone vocabulary file. Default: `None`.
  - `speaker_dict`: speaker id map file. Default: `None`.
  - `spk_id`: Speaker id for multi speaker acoustic model. Default: `0`.
  - `voc`: Vocoder type of tts task. Default: `pwgan_csmsc`.
  - `voc_config`: Config of vocoder. Use deault config when it is None. Default: `None`.
  - `voc_ckpt`: Vocoder checkpoint. Use pretrained model when it is None. Default: `None`.
  - `voc_stat`: Mean and standard deviation used to normalize spectrogram when training vocoder. Default: `None`.
  - `lang`: Language of tts task. Default: `zh`.
  - `device`: Choose device to execute model inference. Default: default device of paddlepaddle in current environment.
  - `output`: Output wave filepath. Default: `output.wav`.

  Output:
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

  Output:
  ```bash
  Wave file has been generated: output.wav
  ```


### 4. Pretrained Models

Here is a list of pretrained models released by PaddleSpeech that can be used by command and python api:

- Acoustic model
  | Model | Language
  | :--- | :---: |
  | speedyspeech_csmsc| zh
  | fastspeech2_csmsc| zh
  | fastspeech2_aishell3| zh
  | fastspeech2_ljspeech| en
  | fastspeech2_vctk| en

- Vocoder
  | Model | Language
  | :--- | :---: |
  | pwgan_csmsc| zh
  | pwgan_aishell3| zh
  | pwgan_ljspeech| en
  | pwgan_vctk| en
  | mb_melgan_csmsc| zh
