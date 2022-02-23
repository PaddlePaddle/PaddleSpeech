([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the voice service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from easy, meduim and hard to install paddlespeech.

### 2. Prepare config File
The configuration file contains the service-related configuration files and the model configuration related to the voice tasks contained in the service. They are all under the `conf` folder. 

The input of  ASR client demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for thisASR client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

### 3. Server Usage
- Command Line (Recommended)

  ```bash
  # start the service
  paddlespeech_server start --config_file ./conf/application.yaml
  ```

  Usage:
  
  ```bash
  paddlespeech_server start --help
  ```
  Arguments:
  - `config_file`: yaml file of the app, defalut: ./conf/application.yaml
  - `log_file`: log file. Default: ./log/paddlespeech.log

  Output:
  ```bash
  [2022-02-23 11:17:32] [INFO] [server.py:64] Started server process [6384]
  INFO:     Waiting for application startup.
  [2022-02-23 11:17:32] [INFO] [on.py:26] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-02-23 11:17:32] [INFO] [on.py:38] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  [2022-02-23 11:17:32] [INFO] [server.py:204] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)

  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/application.yaml", 
      log_file="./log/paddlespeech.log")
  ```

  Output:
  ```bash
  INFO:     Started server process [529]
  [2022-02-23 14:57:56] [INFO] [server.py:64] Started server process [529]
  INFO:     Waiting for application startup.
  [2022-02-23 14:57:56] [INFO] [on.py:26] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-02-23 14:57:56] [INFO] [on.py:38] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  [2022-02-23 14:57:56] [INFO] [server.py:204] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)

  ```


### 4. ASR Client Usage
- Command Line (Recommended)
   ```
   paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
   ```

  Usage:
  
  ```bash
  paddlespeech_client asr --help
  ```
  Arguments:
  - `server_ip`: server ip. Default: 127.0.0.1
  - `port`: server port. Default: 8090
  - `input`(required): Audio file to be recognized.
  - `sample_rate`: Audio ampling rate, default: 16000.
  - `lang`: Language. Default: "zh_cn".
  - `audio_format`: Audio format. Default: "wav".

  Output:
  ```bash
  [2022-02-23 18:11:22,819] [    INFO] - {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '我认为跑步最重要的就是给我带来了身体健康'}}
  [2022-02-23 18:11:22,820] [    INFO] - time cost 0.689145 s.

  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor

  asrclient_executor = ASRClientExecutor()
  asrclient_executor(
      input="./zh.wav",
      server_ip="127.0.0.1",
      port=8090,
      sample_rate=16000,
      lang="zh_cn",
      audio_format="wav")
  ```

  Output:
  ```bash
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '我认为跑步最重要的就是给我带来了身体健康'}}
  time cost 0.604353 s.
  ```
 
### 5. TTS Client Usage
- Command Line (Recommended)
   ```bash
   paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
   ```
     Usage:
  
    ```bash
    paddlespeech_client tts --help
    ```
    Arguments:
    - `server_ip`: server ip. Default: 127.0.0.1
    - `port`: server port. Default: 8090
    - `input`(required): Input text to generate.
    - `spk_id`: Speaker id for multi-speaker text to speech. Default: 0
    - `speed`: Audio speed, the value should be set between 0 and 3. Default: 1.0
    - `volume`: Audio volume, the value should be set between 0 and 3. Default: 1.0
    - `sample_rate`: Sampling rate, choice: [0, 8000, 16000], the default is the same as the model. Default: 0
    - `output`: Output wave filepath. Default: `output.wav`.

    Output:
    ```bash
    [2022-02-23 15:20:37,875] [    INFO] - {'description': 'success.'}
    [2022-02-23 15:20:37,875] [    INFO] - Save synthesized audio successfully on output.wav.
    [2022-02-23 15:20:37,875] [    INFO] - Audio duration: 3.612500 s.
    [2022-02-23 15:20:37,875] [    INFO] - Response time: 0.348050 s.
    [2022-02-23 15:20:37,875] [    INFO] - RTF: 0.096346


    ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import TTSClientExecutor

  ttsclient_executor = TTSClientExecutor()
  ttsclient_executor(
      input="您好，欢迎使用百度飞桨语音合成服务。",
      server_ip="127.0.0.1",
      port=8090,
      spk_id=0,
      speed=1.0,
      volume=1.0,
      sample_rate=0,
      output="./output.wav")
  ```

  Output:
  ```bash
  {'description': 'success.'}
  Save synthesized audio successfully on ./output.wav.
  Audio duration: 3.612500 s.
  Response time: 0.388317 s.
  RTF: 0.107493

  ```


## Pretrained Models
### ASR model
Here is a list of [ASR pretrained models](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/speech_recognition/README.md#4pretrained-models) released by PaddleSpeech, both command line and python interfaces are available:

| Model | Language | Sample Rate
| :--- | :---: | :---: |
| conformer_wenetspeech| zh| 16000
| transformer_librispeech| en| 16000

### TTS model
Here is a list of [TTS pretrained models](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README.md#4-pretrained-models) released by PaddleSpeech, both command line and python interfaces are available:

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

Here is a list of **TTS pretrained static models** released by PaddleSpeech, both command line and python interfaces are available:
- Acoustic model
  | Model | Language
  | :--- | :---: |
  | speedyspeech_csmsc| zh
  | fastspeech2_csmsc| zh

- Vocoder
  | Model | Language
  | :--- | :---: |
  | pwgan_csmsc| zh
  | mb_melgan_csmsc| zh
  | hifigan_csmsc| zh
