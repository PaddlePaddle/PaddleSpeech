([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the voice service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.2.2** or above.
You can choose one way from meduim and hard to install paddlespeech.

### 2. Prepare config File
The configuration file can be found in `conf/application.yaml` .
Among them, `engine_list` indicates the speech engine that will be included in the service to be started, in the format of `<speech task>_<engine type>`.
At present, the speech tasks integrated by the service include: asr (speech recognition), tts (text to sppech) and cls (audio classification).
Currently the engine type supports two forms: python and inference (Paddle Inference)
**Note:** If the service can be started normally in the container, but the client access IP is unreachable, you can try to replace the `host` address in the configuration file with the local IP address.


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
  INFO:     Uvicorn running on http://127.0.0.1:8090 (Press CTRL+C to quit)
  [2022-02-23 11:17:32] [INFO] [server.py:204] Uvicorn running on http://127.0.0.1:8090 (Press CTRL+C to quit)

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
  INFO:     Uvicorn running on http://127.0.0.1:8090 (Press CTRL+C to quit)
  [2022-02-23 14:57:56] [INFO] [server.py:204] Uvicorn running on http://127.0.0.1:8090 (Press CTRL+C to quit)

  ```


### 4. ASR Client Usage
**Note:** The response time will be slightly longer when using the client for the first time
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
  import json

  asrclient_executor = ASRClientExecutor()
  res = asrclient_executor(
      input="./zh.wav",
      server_ip="127.0.0.1",
      port=8090,
      sample_rate=16000,
      lang="zh_cn",
      audio_format="wav")
  print(res.json())
  ```

  Output:
  ```bash
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '我认为跑步最重要的就是给我带来了身体健康'}}
  ```
 
### 5. TTS Client Usage
**Note:** The response time will be slightly longer when using the client for the first time
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
    - `output`: Output wave filepath. Default: None, which means not to save the audio to the local.

    Output:
    ```bash
    [2022-02-23 15:20:37,875] [    INFO] - {'description': 'success.'}
    [2022-02-23 15:20:37,875] [    INFO] - Save synthesized audio successfully on output.wav.
    [2022-02-23 15:20:37,875] [    INFO] - Audio duration: 3.612500 s.
    [2022-02-23 15:20:37,875] [    INFO] - Response time: 0.348050 s.

    ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import TTSClientExecutor
  import json

  ttsclient_executor = TTSClientExecutor()
  res = ttsclient_executor(
      input="您好，欢迎使用百度飞桨语音合成服务。",
      server_ip="127.0.0.1",
      port=8090,
      spk_id=0,
      speed=1.0,
      volume=1.0,
      sample_rate=0,
      output="./output.wav")

  response_dict = res.json()
  print(response_dict["message"])
  print("Save synthesized audio successfully on %s." % (response_dict['result']['save_path']))
  print("Audio duration: %f s." %(response_dict['result']['duration']))
  ```

  Output:
  ```bash
  {'description': 'success.'}
  Save synthesized audio successfully on ./output.wav.
  Audio duration: 3.612500 s.

  ```

### 6. CLS Client Usage
**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)
   ```
   paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
   ```

  Usage:
  
  ```bash
  paddlespeech_client cls --help
  ```
  Arguments:
  - `server_ip`: server ip. Default: 127.0.0.1
  - `port`: server port. Default: 8090
  - `input`(required): Audio file to be classified.
  - `topk`: topk scores of classification result.

  Output:
  ```bash
  [2022-03-09 20:44:39,974] [    INFO] - {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'topk': 1, 'results': [{'class_name': 'Speech', 'prob': 0.9027184844017029}]}}
  [2022-03-09 20:44:39,975] [    INFO] - Response time 0.104360 s.


  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import CLSClientExecutor
  import json

  clsclient_executor = CLSClientExecutor()
  res = clsclient_executor(
      input="./zh.wav",
      server_ip="127.0.0.1",
      port=8090,
      topk=1)
  print(res.json())
  ```

  Output:
  ```bash
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'topk': 1, 'results': [{'class_name': 'Speech', 'prob': 0.9027184844017029}]}}

  ```


## Models supported by the service
### ASR model
Get all models supported by the ASR service via `paddlespeech_server stats --task asr`, where static models can be used for paddle inference inference.

### TTS model
Get all models supported by the TTS service via `paddlespeech_server stats --task tts`, where static models can be used for paddle inference inference.

### CLS model
Get all models supported by the CLS service via `paddlespeech_server stats --task cls`, where static models can be used for paddle inference inference.
