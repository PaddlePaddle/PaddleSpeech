([简体中文](./README_cn.md)|English)

# Streaming Speech Synthesis Service

## Introduction
This demo is an implementation of starting the streaming speech synthesis service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.2.1** or above.
You can choose one way from meduim and hard to install paddlespeech.


### 2. Prepare config File
The configuration file can be found in `conf/tts_online_application.yaml` 。
Among them, `protocol` indicates the network protocol used by the streaming TTS service. Currently, both http and websocket are supported.
`engine_list` indicates the speech engine that will be included in the service to be started, in the format of `<speech task>_<engine type>`.
This demo mainly introduces the streaming speech synthesis service, so the speech task should be set to `tts`.
Currently, the engine type supports two forms: **online**  and **online-onnx**. `online` indicates an engine that uses python for dynamic graph inference; `online-onnx` indicates an engine that uses onnxruntime for inference. The inference speed of online-onnx is faster.
Streaming TTS AM model support: **fastspeech2 and fastspeech2_cnndecoder**; Voc model support: **hifigan and mb_melgan**


### 3. Server Usage
- Command Line (Recommended)

  ```bash
  # start the service
  paddlespeech_server start --config_file ./conf/tts_online_application.yaml
  ```

  Usage:
  
  ```bash
  paddlespeech_server start --help
  ```
  Arguments:
  - `config_file`: yaml file of the app, defalut: ./conf/tts_online_application.yaml
  - `log_file`: log file. Default: ./log/paddlespeech.log

  Output:
  ```bash
  [2022-04-24 20:05:27,887] [    INFO] - The first response time of the 0 warm up: 1.0123658180236816 s
  [2022-04-24 20:05:28,038] [    INFO] - The first response time of the 1 warm up: 0.15108466148376465 s
  [2022-04-24 20:05:28,191] [    INFO] - The first response time of the 2 warm up: 0.15317344665527344 s
  [2022-04-24 20:05:28,192] [    INFO] - **********************************************************************
  INFO:     Started server process [14638]
  [2022-04-24 20:05:28] [INFO] [server.py:75] Started server process [14638]
  INFO:     Waiting for application startup.
  [2022-04-24 20:05:28] [INFO] [on.py:45] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-04-24 20:05:28] [INFO] [on.py:59] Application startup complete.
  INFO:     Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)
  [2022-04-24 20:05:28] [INFO] [server.py:211] Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)

  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/tts_online_application.yaml", 
      log_file="./log/paddlespeech.log")
  ```

  Output:
  ```bash
  [2022-04-24 21:00:16,934] [    INFO] - The first response time of the 0 warm up: 1.268730878829956 s
  [2022-04-24 21:00:17,046] [    INFO] - The first response time of the 1 warm up: 0.11168622970581055 s
  [2022-04-24 21:00:17,151] [    INFO] - The first response time of the 2 warm up: 0.10413002967834473 s
  [2022-04-24 21:00:17,151] [    INFO] - **********************************************************************
  INFO:     Started server process [320]
  [2022-04-24 21:00:17] [INFO] [server.py:75] Started server process [320]
  INFO:     Waiting for application startup.
  [2022-04-24 21:00:17] [INFO] [on.py:45] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-04-24 21:00:17] [INFO] [on.py:59] Application startup complete.
  INFO:     Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)
  [2022-04-24 21:00:17] [INFO] [server.py:211] Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)


  ```

 
### 4. Streaming TTS client Usage
- Command Line (Recommended)

    ```bash
    # Access http streaming TTS service
    paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav

    # Access websocket streaming TTS service
    paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol websocket --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
    ```
    Usage:
  
    ```bash
    paddlespeech_client tts_online --help
    ```

    Arguments:
    - `server_ip`: erver ip. Default: 127.0.0.1
    - `port`: server port. Default: 8092
    - `protocol`: Service protocol, choices: [http, websocket], default: http.
    - `input`: (required): Input text to generate.
    - `spk_id`: Speaker id for multi-speaker text to speech. Default: 0
    - `speed`: Audio speed, the value should be set between 0 and 3. Default: 1.0
    - `volume`: Audio volume, the value should be set between 0 and 3. Default: 1.0
    - `sample_rate`: Sampling rate, choices: [0, 8000, 16000], the default is the same as the model. Default: 0
    - `output`: Output wave filepath. Default: None, which means not to save the audio to the local.
    - `play`: Whether to play audio, play while synthesizing, default value: False, which means not playing. **Playing audio needs to rely on the pyaudio library**.

    
    Output:
    ```bash
    [2022-04-24 21:08:18,559] [    INFO] - tts http client start
    [2022-04-24 21:08:21,702] [    INFO] - 句子：您好，欢迎使用百度飞桨语音合成服务。
    [2022-04-24 21:08:21,703] [    INFO] - 首包响应：0.18863153457641602 s
    [2022-04-24 21:08:21,704] [    INFO] - 尾包响应：3.1427218914031982 s
    [2022-04-24 21:08:21,704] [    INFO] - 音频时长：3.825 s
    [2022-04-24 21:08:21,704] [    INFO] - RTF: 0.8216266382753459
    [2022-04-24 21:08:21,739] [    INFO] - 音频保存至：output.wav

    ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import TTSOnlineClientExecutor
  import json

  executor = TTSOnlineClientExecutor()
  executor(
      input="您好，欢迎使用百度飞桨语音合成服务。",
      server_ip="127.0.0.1",
      port=8092,
      protocol="http",
      spk_id=0,
      speed=1.0,
      volume=1.0,
      sample_rate=0,
      output="./output.wav",
      play=False)

  ```

  Output:
  ```bash
  [2022-04-24 21:11:13,798] [    INFO] - tts http client start
  [2022-04-24 21:11:16,800] [    INFO] - 句子：您好，欢迎使用百度飞桨语音合成服务。
  [2022-04-24 21:11:16,801] [    INFO] - 首包响应：0.18234872817993164 s
  [2022-04-24 21:11:16,801] [    INFO] - 尾包响应：3.0013909339904785 s
  [2022-04-24 21:11:16,802] [    INFO] - 音频时长：3.825 s
  [2022-04-24 21:11:16,802] [    INFO] - RTF: 0.7846773683635238
  [2022-04-24 21:11:16,837] [    INFO] - 音频保存至：./output.wav


  ```

  
