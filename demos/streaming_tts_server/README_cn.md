(简体中文|[English](./README.md))

# 流式语音合成服务

## 介绍
这个demo是一个启动流式语音合成服务和访问该服务的实现。 它可以通过使用`paddlespeech_server` 和 `paddlespeech_client`的单个命令或 python 的几行代码来实现。


## 使用方法
### 1. 安装
请看 [安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

推荐使用 **paddlepaddle 2.2.2** 或以上版本。
你可以从 medium，hard 两种方式中选择一种方式安装 PaddleSpeech。


### 2. 准备配置文件
配置文件可参见 `conf/tts_online_application.yaml` 。
- `protocol` 表示该流式 TTS 服务使用的网络协议，目前支持 **http 和 websocket** 两种。
- `engine_list` 表示即将启动的服务将会包含的语音引擎，格式为 <语音任务>_<引擎类型>。
    - 该 demo 主要介绍流式语音合成服务，因此语音任务应设置为 tts。
    - 目前引擎类型支持两种形式：**online** 表示使用python进行动态图推理的引擎；**online-onnx** 表示使用 onnxruntime 进行推理的引擎。其中，online-onnx 的推理速度更快。
- 流式 TTS 引擎的 AM 模型支持：**fastspeech2 以及fastspeech2_cnndecoder**; Voc 模型支持：**hifigan, mb_melgan**
- 流式 am 推理中，每次会对一个 chunk 的数据进行推理以达到流式的效果。其中 `am_block` 表示 chunk 中的有效帧数，`am_pad` 表示一个 chunk 中 am_block 前后各加的帧数。am_pad 的存在用于消除流式推理产生的误差，避免由流式推理对合成音频质量的影响。
    - fastspeech2 不支持流式 am 推理，因此 am_pad 与 m_block 对它无效
    - fastspeech2_cnndecoder 支持流式推理，当 am_pad=12 时，流式推理合成音频与非流式合成音频一致
- 流式 voc 推理中，每次会对一个 chunk 的数据进行推理以达到流式的效果。其中 `voc_block` 表示chunk中的有效帧数，`voc_pad` 表示一个 chunk 中 voc_block 前后各加的帧数。voc_pad 的存在用于消除流式推理产生的误差，避免由流式推理对合成音频质量的影响。
    - hifigan, mb_melgan 均支持流式 voc 推理
    - 当 voc 模型为 mb_melgan，当 voc_pad=14 时，流式推理合成音频与非流式合成音频一致；voc_pad 最小可以设置为7，合成音频听感上没有异常，若 voc_pad 小于7，合成音频听感上存在异常。
    - 当 voc 模型为 hifigan，当 voc_pad=20 时，流式推理合成音频与非流式合成音频一致；当 voc_pad=14 时，合成音频听感上没有异常。
- 推理速度：mb_melgan > hifigan; 音频质量：mb_melgan < hifigan
- **注意：** 如果在容器里可正常启动服务，但客户端访问 ip 不可达，可尝试将配置文件中 `host` 地址换成本地 ip 地址。


### 3. 使用http协议的流式语音合成服务端及客户端使用方法
#### 3.1 服务端使用方法
- 命令行 (推荐使用)

  启动服务（配置文件默认使用http）：
  ```bash
  paddlespeech_server start --config_file ./conf/tts_online_application.yaml
  ```

  使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  参数:
  - `config_file`: 服务的配置文件，默认： ./conf/tts_online_application.yaml
  - `log_file`: log 文件. 默认：./log/paddlespeech.log

  输出:
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

  输出：
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

#### 3.2 客户端使用方法
- 命令行 (推荐使用)

    访问 http 流式TTS服务：

    ```bash
    paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol http --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
    ```

    使用帮助:
  
    ```bash
    paddlespeech_client tts_online --help
    ```

    参数:
    - `server_ip`: 服务端ip地址，默认: 127.0.0.1。
    - `port`: 服务端口，默认: 8092。
    - `protocol`: 服务协议，可选 [http, websocket], 默认: http。
    - `input`: (必须输入): 待合成的文本。
    - `spk_id`: 说话人 id，用于多说话人语音合成，默认值： 0。
    - `speed`: 音频速度，该值应设置在 0 到 3 之间。 默认值：1.0
    - `volume`: 音频音量，该值应设置在 0 到 3 之间。 默认值： 1.0
    - `sample_rate`: 采样率，可选 [0, 8000, 16000]，默认值：0，表示与模型采样率相同
    - `output`: 输出音频的路径， 默认值：None，表示不保存音频到本地。
    - `play`: 是否播放音频，边合成边播放， 默认值：False，表示不播放。**播放音频需要依赖pyaudio库**。
    - `spk_id, speed, volume, sample_rate` 在流式语音合成服务中暂时不生效。

    
    输出:
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

  输出:
  ```bash
  [2022-04-24 21:11:13,798] [    INFO] - tts http client start
  [2022-04-24 21:11:16,800] [    INFO] - 句子：您好，欢迎使用百度飞桨语音合成服务。
  [2022-04-24 21:11:16,801] [    INFO] - 首包响应：0.18234872817993164 s
  [2022-04-24 21:11:16,801] [    INFO] - 尾包响应：3.0013909339904785 s
  [2022-04-24 21:11:16,802] [    INFO] - 音频时长：3.825 s
  [2022-04-24 21:11:16,802] [    INFO] - RTF: 0.7846773683635238
  [2022-04-24 21:11:16,837] [    INFO] - 音频保存至：./output.wav
  ```

 
### 4. 使用websocket协议的流式语音合成服务端及客户端使用方法
#### 4.1 服务端使用方法
- 命令行 (推荐使用)
  首先修改配置文件 `conf/tts_online_application.yaml`， **将 `protocol` 设置为 `websocket`**。
  启动服务：
  ```bash
  paddlespeech_server start --config_file ./conf/tts_online_application.yaml
  ```

  使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  参数:
  - `config_file`: 服务的配置文件，默认： ./conf/tts_online_application.yaml
  - `log_file`: log 文件. 默认：./log/paddlespeech.log

  输出:
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

  输出：
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

#### 4.2 客户端使用方法
- 命令行 (推荐使用)

    访问 websocket 流式TTS服务：

    ```bash
    paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol websocket --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav
    ```

    使用帮助:
  
    ```bash
    paddlespeech_client tts_online --help
    ```

    参数:
    - `server_ip`: 服务端ip地址，默认: 127.0.0.1。
    - `port`: 服务端口，默认: 8092。
    - `protocol`: 服务协议，可选 [http, websocket], 默认: http。
    - `input`: (必须输入): 待合成的文本。
    - `spk_id`: 说话人 id，用于多说话人语音合成，默认值： 0。
    - `speed`: 音频速度，该值应设置在 0 到 3 之间。 默认值：1.0
    - `volume`: 音频音量，该值应设置在 0 到 3 之间。 默认值： 1.0
    - `sample_rate`: 采样率，可选 [0, 8000, 16000]，默认值：0，表示与模型采样率相同
    - `output`: 输出音频的路径， 默认值：None，表示不保存音频到本地。
    - `play`: 是否播放音频，边合成边播放， 默认值：False，表示不播放。**播放音频需要依赖pyaudio库**。
    - `spk_id, speed, volume, sample_rate` 在流式语音合成服务中暂时不生效。

    
    输出:
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

  输出:
  ```bash
    [2022-04-27 10:22:48,852] [    INFO] - tts websocket client start
    [2022-04-27 10:22:49,080] [    INFO] - 句子：您好，欢迎使用百度飞桨语音合成服务。
    [2022-04-27 10:22:49,080] [    INFO] - 首包响应：0.21017956733703613 s
    [2022-04-27 10:22:52,100] [    INFO] - 尾包响应：3.2304444313049316 s
    [2022-04-27 10:22:52,101] [    INFO] - 音频时长：3.825 s
    [2022-04-27 10:22:52,101] [    INFO] - RTF: 0.8445606356352762
    [2022-04-27 10:22:52,134] [    INFO] - 音频保存至：./output.wav

  ```


  
