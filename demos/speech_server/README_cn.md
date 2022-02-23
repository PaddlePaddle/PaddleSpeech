([简体中文](./README_cn.md)|English)

# 语音服务

## 介绍
这个demo是一个启动语音服务和访问服务的实现。 它可以通过使用`paddlespeech_server` 和 `paddlespeech_client`的单个命令或 python 的几行代码来实现。


## 使用方法
### 1. 安装
请看 [安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

你可以从 easy，medium，hard 三中方式中选择一种方式安装 PaddleSpeech。

### 2. 准备配置文件
配置文件包含服务相关的配置文件和服务中包含的语音任务相关的模型配置。 它们都在 `conf` 文件夹下。

### 3. 服务端使用方法
- 命令行 (推荐使用)

  ```bash
  # 启动服务
  paddlespeech_server start --config_file ./conf/application.yaml
  ```

  使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  参数:
  - `config_file`: 服务的配置文件，默认： ./conf/application.yaml
  - `log_file`: log 文件. 默认：./log/paddlespeech.log

  输出:
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

  输出：
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

### 4. ASR客户端使用方法
- 命令行 (推荐使用)
   ```
   paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input ./paddlespeech/server/tests/16_audio.wav
   ```

    使用帮助:
  
    ```bash
    paddlespeech_client asr --help
    ```

    参数:
    - `server_ip`: 服务端ip地址，默认: 127.0.0.1。
    - `port`: 服务端口，默认: 8090。
    - `input`(必须输入): 用于识别的音频文件。
    - `sample_rate`: 音频采样率，默认值：16000。
    - `lang`: 模型语言，默认值：zh_cn。
    - `audio_format`: 音频格式，默认值：wav。

    输出:

    ```bash
    [2022-02-23 11:19:45,646] [    INFO] - {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '广州医生跑北马中断比赛就心跳骤停者'}}
    [2022-02-23 11:19:45,646] [    INFO] - time cost 0.659491 s.

    ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor

  asrclient_executor = ASRClientExecutor()
  asrclient_executor(
      input="./16_audio.wav",
      server_ip="127.0.0.1",
      port=8090,
      sample_rate=16000,
      lang="zh_cn",
      audio_format="wav")
  ```

  输出:
  ```bash
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '广州医生跑北马中断比赛就心跳骤停者'}}
  time cost 0.802639 s.

  ```
 
### 5. TTS客户端使用方法
   ```bash
   paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
   ```
    使用帮助:
  
    ```bash
    paddlespeech_client tts --help
    ```

    参数:
    - `server_ip`: 服务端ip地址，默认: 127.0.0.1。
    - `port`: 服务端口，默认: 8090。
    - `input`(必须输入): 待合成的文本。
    - `spk_id`: 说话人 id，用于多说话人语音合成，默认值： 0。
    - `speed`: 音频速度，该值应设置在 0 到 3 之间。 默认值：1.0
    - `volume`: 音频音量，该值应设置在 0 到 3 之间。 默认值： 1.0
    - `sample_rate`: 采样率，可选 [0, 8000, 16000]，默认与模型相同。 默认值：0
    - `output`: 输出音频的路径， 默认值：output.wav。

    输出:
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

  输出:
  ```bash
  {'description': 'success.'}
  Save synthesized audio successfully on ./output.wav.
  Audio duration: 3.612500 s.
  Response time: 0.388317 s.
  RTF: 0.107493

  ```

## Pretrained Models
### ASR model
下面是PaddleSpeech发布的[ASR预训练模型](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/speech_recognition/README.md#4pretrained-models)列表，命令行和python接口均可用：

| Model | Language | Sample Rate
| :--- | :---: | :---: |
| conformer_wenetspeech| zh| 16000
| transformer_librispeech| en| 16000

### TTS model
下面是PaddleSpeech发布的 [TTS预训练模型](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README.md#4-pretrained-models) 列表，命令行和python接口均可用：

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

下面是PaddleSpeech发布的 **TTS预训练静态模型** 列表，命令行和python接口均可用：
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