([简体中文](./README_cn.md)|English)

# Streaming Speech Synthesis Service

## Introduction
This demo is an implementation of starting the streaming speech synthesis service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.2.2** or above.
You can choose one way from meduim and hard to install paddlespeech.


### 2. Prepare config File
The configuration file can be found in `conf/tts_online_application.yaml`.
- `protocol` indicates the network protocol used by the streaming TTS service. Currently, both **http and websocket** are supported.
- `engine_list` indicates the speech engine that will be included in the service to be started, in the format of `<speech task>_<engine type>`.
    - This demo mainly introduces the streaming speech synthesis service, so the speech task should be set to `tts`.
    - the engine type supports two forms: **online**  and **online-onnx**. `online` indicates an engine that uses python for dynamic graph inference; `online-onnx` indicates an engine that uses onnxruntime for inference. The inference speed of online-onnx is faster.
- Streaming TTS engine AM model support: **fastspeech2 and fastspeech2_cnndecoder**; Voc model support: **hifigan and mb_melgan**
- In streaming am inference, one chunk of data is inferred at a time to achieve a streaming effect. Among them, `am_block` indicates the number of valid frames in the chunk, and `am_pad` indicates the number of frames added before and after am_block in a chunk. The existence of am_pad is used to eliminate errors caused by streaming inference and avoid the influence of streaming inference on the quality of synthesized audio.
    - fastspeech2 does not support streaming am inference, so am_pad and am_block have no effect on it.
    - fastspeech2_cnndecoder supports streaming inference. When am_pad=12, streaming inference synthesized audio is consistent with non-streaming synthesized audio.
- In streaming voc inference, one chunk of data is inferred at a time to achieve a streaming effect. Where `voc_block` indicates the number of valid frames in the chunk, and `voc_pad` indicates the number of frames added before and after the voc_block in a chunk. The existence of voc_pad is used to eliminate errors caused by streaming inference and avoid the influence of streaming inference on the quality of synthesized audio.
    - Both hifigan and mb_melgan support streaming voc inference.
    - When the voc model is mb_melgan, when voc_pad=14, the synthetic audio for streaming inference is consistent with the non-streaming synthetic audio; the minimum voc_pad can be set to 7, and the synthetic audio has no abnormal hearing. If the voc_pad is less than 7, the synthetic audio sounds abnormal.
    - When the voc model is hifigan, when voc_pad=20, the streaming inference synthetic audio is consistent with the non-streaming synthetic audio; when voc_pad=14, the synthetic audio has no abnormal hearing.
- Inference speed: mb_melgan > hifigan; Audio quality: mb_melgan < hifigan
- **Note:** If the service can be started normally in the container, but the client access IP is unreachable, you can try to replace the `host` address in the configuration file with the local IP address.



### 3. Streaming speech synthesis server and client using http protocol
#### 3.1 Server Usage
- Command Line (Recommended)

  Start the service (the configuration file uses http by default):
  ```bash
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

#### 3.2 Streaming TTS client Usage
- Command Line (Recommended)

    Access http streaming TTS service:

    ```bash
    paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol http --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
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
    - `spk_id, speed, volume, sample_rate` do not take effect in streaming speech synthesis service temporarily.
    
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

 
### 4. Streaming speech synthesis server and client using websocket protocol
#### 4.1 Server Usage
- Command Line (Recommended)
  First modify the configuration file `conf/tts_online_application.yaml`, **set `protocol` to `websocket`**.
  Start the service:
  ```bash
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
    [2022-04-27 10:18:09,107] [    INFO] - The first response time of the 0 warm up: 1.1551103591918945 s
    [2022-04-27 10:18:09,219] [    INFO] - The first response time of the 1 warm up: 0.11204338073730469 s
    [2022-04-27 10:18:09,324] [    INFO] - The first response time of the 2 warm up: 0.1051797866821289 s
    [2022-04-27 10:18:09,325] [    INFO] - **********************************************************************
    INFO:     Started server process [17600]
    [2022-04-27 10:18:09] [INFO] [server.py:75] Started server process [17600]
    INFO:     Waiting for application startup.
    [2022-04-27 10:18:09] [INFO] [on.py:45] Waiting for application startup.
    INFO:     Application startup complete.
    [2022-04-27 10:18:09] [INFO] [on.py:59] Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)
    [2022-04-27 10:18:09] [INFO] [server.py:211] Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)


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
    [2022-04-27 10:20:16,660] [    INFO] - The first response time of the 0 warm up: 1.0945196151733398 s
    [2022-04-27 10:20:16,773] [    INFO] - The first response time of the 1 warm up: 0.11222052574157715 s
    [2022-04-27 10:20:16,878] [    INFO] - The first response time of the 2 warm up: 0.10494542121887207 s
    [2022-04-27 10:20:16,878] [    INFO] - **********************************************************************
    INFO:     Started server process [23466]
    [2022-04-27 10:20:16] [INFO] [server.py:75] Started server process [23466]
    INFO:     Waiting for application startup.
    [2022-04-27 10:20:16] [INFO] [on.py:45] Waiting for application startup.
    INFO:     Application startup complete.
    [2022-04-27 10:20:16] [INFO] [on.py:59] Application startup complete.
    INFO:     Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)
    [2022-04-27 10:20:16] [INFO] [server.py:211] Uvicorn running on http://127.0.0.1:8092 (Press CTRL+C to quit)

  ```

#### 4.2 Streaming TTS client Usage
- Command Line (Recommended)

    Access websocket streaming TTS service:

    ```bash
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
    - `spk_id, speed, volume, sample_rate` do not take effect in streaming speech synthesis service temporarily.

    
    Output:
    ```bash
    [2022-04-27 10:21:04,262] [    INFO] - tts websocket client start
    [2022-04-27 10:21:04,496] [    INFO] - 句子：您好，欢迎使用百度飞桨语音合成服务。
    [2022-04-27 10:21:04,496] [    INFO] - 首包响应：0.2124948501586914 s
    [2022-04-27 10:21:07,483] [    INFO] - 尾包响应：3.199106454849243 s
    [2022-04-27 10:21:07,484] [    INFO] - 音频时长：3.825 s
    [2022-04-27 10:21:07,484] [    INFO] - RTF: 0.8363677006141812
    [2022-04-27 10:21:07,516] [    INFO] - 音频保存至：output.wav

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
      protocol="websocket",
      spk_id=0,
      speed=1.0,
      volume=1.0,
      sample_rate=0,
      output="./output.wav",
      play=False)

  ```

  Output:
  ```bash
    [2022-04-27 10:22:48,852] [    INFO] - tts websocket client start
    [2022-04-27 10:22:49,080] [    INFO] - 句子：您好，欢迎使用百度飞桨语音合成服务。
    [2022-04-27 10:22:49,080] [    INFO] - 首包响应：0.21017956733703613 s
    [2022-04-27 10:22:52,100] [    INFO] - 尾包响应：3.2304444313049316 s
    [2022-04-27 10:22:52,101] [    INFO] - 音频时长：3.825 s
    [2022-04-27 10:22:52,101] [    INFO] - RTF: 0.8445606356352762
    [2022-04-27 10:22:52,134] [    INFO] - 音频保存至：./output.wav

  ```



  
