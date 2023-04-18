([English](./README.md)|中文)

# 流式语音识别服务

## 介绍
这个 demo 是一个启动流式语音服务和访问服务的实现。 它可以通过使用 `paddlespeech_server` 和 `paddlespeech_client` 的单个命令或 python 的几行代码来实现。

**流式语音识别服务只支持 `weboscket` 协议，不支持 `http` 协议。**

服务接口定义请参考:
- [PaddleSpeech Streaming Server WebSocket API](https://github.com/PaddlePaddle/PaddleSpeech/wiki/PaddleSpeech-Server-WebSocket-API)

## 使用方法
### 1. 安装
安装 PaddleSpeech 的详细过程请看 [安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md)。

推荐使用 **paddlepaddle 2.4rc** 或以上版本。

你可以从简单，中等，困难 几种方式中选择一种方式安装 PaddleSpeech。

**如果使用简单模式安装，需要自行准备 yaml 文件，可参考 conf 目录下的 yaml 文件。**

### 2. 准备配置文件

流式ASR的服务启动脚本和服务测试脚本存放在 `PaddleSpeech/demos/streaming_asr_server` 目录。
下载好 `PaddleSpeech` 之后，进入到 `PaddleSpeech/demos/streaming_asr_server` 目录。
配置文件可参见该目录下 `conf/ws_application.yaml` 和 `conf/ws_conformer_wenetspeech_application.yaml` 。

目前服务集成的模型有： DeepSpeech2 和 conformer模型，对应的配置文件如下：
* DeepSpeech: `conf/ws_application.yaml`
* conformer: `conf/ws_conformer_wenetspeech_application.yaml`


这个 ASR client 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 ASR client的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 3. 服务端使用方法
- 命令行 (推荐使用)
  **注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数部署在 `gpu` 上。
  ```bash
  # 在 PaddleSpeech/demos/streaming_asr_server 目录启动服务
  paddlespeech_server start --config_file ./conf/ws_conformer_wenetspeech_application.yaml
  # 你如果愿意为了增加解码的速度而牺牲一定的模型精度，你可以使用如下的脚本 
   paddlespeech_server start --config_file ./conf/ws_conformer_wenetspeech_application_faster.yaml
  ```

  使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  参数:
  - `config_file`: 服务的配置文件，默认： `./conf/application.yaml`
  - `log_file`: log 文件. 默认：`./log/paddlespeech.log`

  输出:
  ```text
  [2022-05-14 04:56:13,086] [    INFO] - create the online asr engine instance
  [2022-05-14 04:56:13,086] [    INFO] - paddlespeech_server set the device: cpu
  [2022-05-14 04:56:13,087] [    INFO] - Load the pretrained model, tag = conformer_online_wenetspeech-zh-16k
  [2022-05-14 04:56:13,087] [    INFO] - File /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz md5        checking...
  [2022-05-14 04:56:17,542] [    INFO] - Use pretrained model stored in: /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.  0.0a.model.tar
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar/model.yaml
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar/exp/               chunk_conformer/checkpoints/avg_10.pdparams
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar/exp/               chunk_conformer/checkpoints/avg_10.pdparams
  [2022-05-14 04:56:17,852] [    INFO] - start to create the stream conformer asr engine
  [2022-05-14 04:56:17,863] [    INFO] - model name: conformer_online
  [2022-05-14 04:56:22,756] [    INFO] - create the transformer like model success
  [2022-05-14 04:56:22,758] [    INFO] - Initialize ASR server engine successfully.
  INFO:     Started server process [4242]
  [2022-05-14 04:56:22] [INFO] [server.py:75] Started server process [4242]
  INFO:     Waiting for application startup.
  [2022-05-14 04:56:22] [INFO] [on.py:45] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-05-14 04:56:22] [INFO] [on.py:59] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  [2022-05-14 04:56:22] [INFO] [server.py:211] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  ```

- Python API
  **注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数部署在 `gpu` 上。
  ```python
  # 在 PaddleSpeech/demos/streaming_asr_server 目录
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/ws_conformer_wenetspeech_application_faster.yaml", 
      log_file="./log/paddlespeech.log")
  ```

  输出:
  ```text
  [2022-05-14 04:56:13,086] [    INFO] - create the online asr engine instance
  [2022-05-14 04:56:13,086] [    INFO] - paddlespeech_server set the device: cpu
  [2022-05-14 04:56:13,087] [    INFO] - Load the pretrained model, tag = conformer_online_wenetspeech-zh-16k
  [2022-05-14 04:56:13,087] [    INFO] - File /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz md5        checking...
  [2022-05-14 04:56:17,542] [    INFO] - Use pretrained model stored in: /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.  0.0a.model.tar
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar/model.yaml
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar/exp/               chunk_conformer/checkpoints/avg_10.pdparams
  [2022-05-14 04:56:17,543] [    INFO] - /root/.paddlespeech/models/conformer_online_wenetspeech-zh-16k/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar/exp/               chunk_conformer/checkpoints/avg_10.pdparams
  [2022-05-14 04:56:17,852] [    INFO] - start to create the stream conformer asr engine
  [2022-05-14 04:56:17,863] [    INFO] - model name: conformer_online
  [2022-05-14 04:56:22,756] [    INFO] - create the transformer like model success
  [2022-05-14 04:56:22,758] [    INFO] - Initialize ASR server engine successfully.
  INFO:     Started server process [4242]
  [2022-05-14 04:56:22] [INFO] [server.py:75] Started server process [4242]
  INFO:     Waiting for application startup.
  [2022-05-14 04:56:22] [INFO] [on.py:45] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-05-14 04:56:22] [INFO] [on.py:59] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  [2022-05-14 04:56:22] [INFO] [server.py:211] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  ```

### 4. ASR 客户端使用方法

**注意：** 初次使用客户端时响应时间会略长
- 命令行 (推荐使用)

  若 `127.0.0.1` 不能访问，则需要使用实际服务 IP 地址

  ```bash
  paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
  ```

  使用帮助:

  ```bash
  paddlespeech_client asr_online --help
  ```

  参数:
  - `server_ip`: 服务端ip地址，默认: 127.0.0.1。
  - `port`: 服务端口，默认: 8090。
  - `input`(必须输入): 用于识别的音频文件。
  - `sample_rate`: 音频采样率，默认值：16000。
  - `lang`: 模型语言，默认值：zh_cn。
  - `audio_format`: 音频格式，默认值：wav。
  - `punc.server_ip` 标点预测服务的ip。默认是None。
  - `punc.server_port` 标点预测服务的端口port。默认是None。

  输出:
  ```text
  [2022-05-06 21:10:35,598] [    INFO] - Start to do streaming asr client
  [2022-05-06 21:10:35,600] [    INFO] - asr websocket client start
  [2022-05-06 21:10:35,600] [    INFO] - endpoint: ws://127.0.0.1:8390/paddlespeech/asr/streaming
  [2022-05-06 21:10:35,600] [    INFO] - start to process the wavscp: ./zh.wav
  [2022-05-06 21:10:35,670] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
  [2022-05-06 21:10:35,699] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,713] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,726] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,738] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,750] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,762] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,774] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:35,786] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,387] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,398] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,407] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,416] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,425] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,434] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,442] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:10:36,930] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,938] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,946] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,954] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,962] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,970] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,977] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:36,985] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:10:37,484] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:37,492] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:37,500] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:37,508] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:37,517] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:37,525] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:37,532] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:10:38,050] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,058] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,066] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,073] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,081] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,089] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,097] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,105] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:10:38,630] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:38,639] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:38,647] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:38,655] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:38,663] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:38,671] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:38,679] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:10:39,216] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,224] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,232] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,240] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,248] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,256] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,264] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,272] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:10:39,885] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:10:39,896] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:10:39,905] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:10:39,915] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:10:39,924] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:10:39,934] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:10:44,827] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '我认为跑步最重要的就是给我带来了身体健康', 'times': [{'w': '我', 'bg': 0.0, 'ed': 0.7000000000000001}, {'w': '认', 'bg': 0.7000000000000001, 'ed': 0.84}, {'w': '为', 'bg': 0.84, 'ed': 1.0}, {'w': '跑', 'bg': 1.0, 'ed': 1.18}, {'w': '步', 'bg': 1.18, 'ed': 1.36}, {'w': '最', 'bg': 1.36, 'ed': 1.5}, {'w': '重', 'bg': 1.5, 'ed': 1.6400000000000001}, {'w': '要', 'bg': 1.6400000000000001, 'ed': 1.78}, {'w': '的', 'bg': 1.78, 'ed': 1.9000000000000001}, {'w': '就', 'bg': 1.9000000000000001, 'ed': 2.06}, {'w': '是', 'bg': 2.06, 'ed': 2.62}, {'w': '给', 'bg': 2.62, 'ed': 3.16}, {'w': '我', 'bg': 3.16, 'ed': 3.3200000000000003}, {'w': '带', 'bg': 3.3200000000000003, 'ed': 3.48}, {'w': '来', 'bg': 3.48, 'ed': 3.62}, {'w': '了', 'bg': 3.62, 'ed': 3.7600000000000002}, {'w': '身', 'bg': 3.7600000000000002, 'ed': 3.9}, {'w': '体', 'bg': 3.9, 'ed': 4.0600000000000005}, {'w': '健', 'bg': 4.0600000000000005, 'ed': 4.26}, {'w': '康', 'bg': 4.26, 'ed': 4.96}]}
  [2022-05-06 21:10:44,827] [    INFO] - audio duration: 4.9968125, elapsed time: 9.225094079971313, RTF=1.846195765794957
  [2022-05-06 21:10:44,828] [    INFO] - asr websocket client finished : 我认为跑步最重要的就是给我带来了身体健康
    ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import ASROnlineClientExecutor

  asrclient_executor = ASROnlineClientExecutor()
  res = asrclient_executor(
      input="./zh.wav",
      server_ip="127.0.0.1",
      port=8090,
      sample_rate=16000,
      lang="zh_cn",
      audio_format="wav")
  print(res)
  ```

  输出:
  ```text
  [2022-05-06 21:14:03,137] [    INFO] - asr websocket client start
  [2022-05-06 21:14:03,137] [    INFO] - endpoint: ws://127.0.0.1:8390/paddlespeech/asr/streaming
  [2022-05-06 21:14:03,149] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
  [2022-05-06 21:14:03,167] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,181] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,194] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,207] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,219] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,230] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,241] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,252] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,768] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,776] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,784] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,792] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,800] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,807] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:03,815] [    INFO] - client receive msg={'result': ''}
  [2022-05-06 21:14:04,301] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,309] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,317] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,325] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,333] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,341] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,349] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,356] [    INFO] - client receive msg={'result': '我认为跑'}
  [2022-05-06 21:14:04,855] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:04,864] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:04,871] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:04,879] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:04,887] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:04,894] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:04,902] [    INFO] - client receive msg={'result': '我认为跑步最重要的'}
  [2022-05-06 21:14:05,418] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,426] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,434] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,442] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,449] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,457] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,465] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,473] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是'}
  [2022-05-06 21:14:05,996] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,006] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,013] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,021] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,029] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,037] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,045] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给'}
  [2022-05-06 21:14:06,581] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,589] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,597] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,605] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,613] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,621] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,628] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:06,636] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了'}
  [2022-05-06 21:14:07,188] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:14:07,196] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:14:07,203] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:14:07,211] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:14:07,219] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:14:07,226] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康'}
  [2022-05-06 21:14:12,158] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '我认为跑步最重要的就是给我带来了身体健康', 'times': [{'w': '我', 'bg': 0.0, 'ed': 0.7000000000000001}, {'w': '认', 'bg': 0.7000000000000001, 'ed': 0.84}, {'w': '为', 'bg': 0.84, 'ed': 1.0}, {'w': '跑', 'bg': 1.0, 'ed': 1.18}, {'w': '步', 'bg': 1.18, 'ed': 1.36}, {'w': '最', 'bg': 1.36, 'ed': 1.5}, {'w': '重', 'bg': 1.5, 'ed': 1.6400000000000001}, {'w': '要', 'bg': 1.6400000000000001, 'ed': 1.78}, {'w': '的', 'bg': 1.78, 'ed': 1.9000000000000001}, {'w': '就', 'bg': 1.9000000000000001, 'ed': 2.06}, {'w': '是', 'bg': 2.06, 'ed': 2.62}, {'w': '给', 'bg': 2.62, 'ed': 3.16}, {'w': '我', 'bg': 3.16, 'ed': 3.3200000000000003}, {'w': '带', 'bg': 3.3200000000000003, 'ed': 3.48}, {'w': '来', 'bg': 3.48, 'ed': 3.62}, {'w': '了', 'bg': 3.62, 'ed': 3.7600000000000002}, {'w': '身', 'bg': 3.7600000000000002, 'ed': 3.9}, {'w': '体', 'bg': 3.9, 'ed': 4.0600000000000005}, {'w': '健', 'bg': 4.0600000000000005, 'ed': 4.26}, {'w': '康', 'bg': 4.26, 'ed': 4.96}]}
  [2022-05-06 21:14:12,159] [    INFO] - audio duration: 4.9968125, elapsed time: 9.019973039627075, RTF=1.8051453881103354
  [2022-05-06 21:14:12,160] [    INFO] - asr websocket client finished
  ```
## 标点预测

### 1. 服务端使用方法

- 命令行
  **注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数部署在 `gpu` 上。
  ```bash
  # 在 PaddleSpeech/demos/streaming_asr_server 目录下启动标点预测服务
  paddlespeech_server start --config_file conf/punc_application.yaml
  ```

  使用方法:
  ```bash
  paddlespeech_server start --help
  ```
  
  参数:
  - `config_file`: 服务的配置文件。
  - `log_file`: log 文件。


  输出:
  ```text
  [2022-05-02 17:59:26,285] [    INFO] - Create the TextEngine Instance
  [2022-05-02 17:59:26,285] [    INFO] - Init the text engine
  [2022-05-02 17:59:26,285] [    INFO] - Text Engine set the device: gpu:0
  [2022-05-02 17:59:26,286] [    INFO] - File /home/users/xiongxinlei/.paddlespeech/models/ernie_linear_p3_wudao-punc-zh/ernie_linear_p3_wudao-punc-zh.tar.gz md5 checking...
  [2022-05-02 17:59:30,810] [    INFO] - Use pretrained model stored in: /home/users/xiongxinlei/.paddlespeech/models/ernie_linear_p3_wudao-punc-zh/ernie_linear_p3_wudao-punc-zh.tar
  W0502 17:59:31.486552  9595 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 10.2, Runtime API Version: 10.2
  W0502 17:59:31.491360  9595 device_context.cc:465] device: 0, cuDNN Version: 7.6.
  [2022-05-02 17:59:34,688] [    INFO] - Already cached /home/users/xiongxinlei/.paddlenlp/models/ernie-1.0/vocab.txt
  [2022-05-02 17:59:34,701] [    INFO] - Init the text engine successfully
  INFO:     Started server process [9595]
  [2022-05-02 17:59:34] [INFO] [server.py:75] Started server process [9595]
  INFO:     Waiting for application startup.
  [2022-05-02 17:59:34] [INFO] [on.py:45] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-05-02 17:59:34] [INFO] [on.py:59] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8190 (Press CTRL+C to quit)
  [2022-05-02 17:59:34] [INFO] [server.py:206] Uvicorn running on http://0.0.0.0:8190 (Press CTRL+C to quit)
  ```

- Python API
  **注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数部署在 `gpu` 上。
  ```python
  # 在 PaddleSpeech/demos/streaming_asr_server 目录
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/punc_application.yaml", 
      log_file="./log/paddlespeech.log")
  ```

  输出:
  ```text
  [2022-05-02 18:09:02,542] [    INFO] - Create the TextEngine Instance
  [2022-05-02 18:09:02,543] [    INFO] - Init the text engine
  [2022-05-02 18:09:02,543] [    INFO] - Text Engine set the device: gpu:0
  [2022-05-02 18:09:02,545] [    INFO] - File /home/users/xiongxinlei/.paddlespeech/models/ernie_linear_p3_wudao-punc-zh/ernie_linear_p3_wudao-punc-zh.tar.gz md5 checking...
  [2022-05-02 18:09:06,919] [    INFO] - Use pretrained model stored in: /home/users/xiongxinlei/.paddlespeech/models/ernie_linear_p3_wudao-punc-zh/ernie_linear_p3_wudao-punc-zh.tar
  W0502 18:09:07.523002 22615 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 10.2, Runtime API Version: 10.2
  W0502 18:09:07.527882 22615 device_context.cc:465] device: 0, cuDNN Version: 7.6.
  [2022-05-02 18:09:10,900] [    INFO] - Already cached /home/users/xiongxinlei/.paddlenlp/models/ernie-1.0/vocab.txt
  [2022-05-02 18:09:10,913] [    INFO] - Init the text engine successfully
  INFO:     Started server process [22615]
  [2022-05-02 18:09:10] [INFO] [server.py:75] Started server process [22615]
  INFO:     Waiting for application startup.
  [2022-05-02 18:09:10] [INFO] [on.py:45] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-05-02 18:09:10] [INFO] [on.py:59] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8190 (Press CTRL+C to quit)
  [2022-05-02 18:09:10] [INFO] [server.py:206] Uvicorn running on http://0.0.0.0:8190 (Press CTRL+C to quit)
  ```

### 2. 标点预测客户端使用方法
**注意：** 初次使用客户端时响应时间会略长

- 命令行 (推荐使用)

  若 `127.0.0.1` 不能访问，则需要使用实际服务 IP 地址

  ```bash
  paddlespeech_client text --server_ip 127.0.0.1 --port 8190 --input "我认为跑步最重要的就是给我带来了身体健康"
  ```
  
  输出:
  ```text
  [2022-05-02 18:12:29,767] [    INFO] - The punc text: 我认为跑步最重要的就是给我带来了身体健康。
  [2022-05-02 18:12:29,767] [    INFO] - Response time 0.096548 s.
  ```

- Python API

  ```python
  from paddlespeech.server.bin.paddlespeech_client import TextClientExecutor

  textclient_executor = TextClientExecutor()
  res = textclient_executor(
      input="我认为跑步最重要的就是给我带来了身体健康",
      server_ip="127.0.0.1",
      port=8190,)
  print(res)
  ```

  输出:
  ```text
  我认为跑步最重要的就是给我带来了身体健康。
  ```

## 联合流式语音识别和标点预测
**注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数将语音识别和标点预测部署在不同的 `gpu` 上。

使用 `streaming_asr_server.py` 和 `punc_server.py` 两个服务，分别启动流式语音识别和标点预测服务。调用 `websocket_client.py` 脚本可以同时调用流式语音识别和标点预测服务。

### 1. 启动服务

```bash
注意：流式语音识别和标点预测通过配置文件配置到不同的显卡上
bash server.sh
```

### 2. 调用服务
- 使用命令行：

  若 `127.0.0.1` 不能访问，则需要使用实际服务 IP 地址

  ```bash
  paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --input ./zh.wav
  ```
  输出:
  ```text
  [2022-05-07 11:21:47,060] [    INFO] - asr websocket client start
  [2022-05-07 11:21:47,060] [    INFO] - endpoint: ws://127.0.0.1:8490/paddlespeech/asr/streaming
  [2022-05-07 11:21:47,080] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
  [2022-05-07 11:21:47,096] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,108] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,120] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,131] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,142] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,152] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,163] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,173] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,705] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,713] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,721] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,728] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,736] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,743] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:47,751] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:21:48,459] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:48,572] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:48,681] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:48,790] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:48,898] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:49,005] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:49,112] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:49,219] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:21:49,935] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:50,062] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:50,186] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:50,310] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:50,435] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:50,560] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:50,686] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:21:51,444] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:51,606] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:51,744] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:51,882] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:52,020] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:52,159] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:52,298] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:52,437] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:21:53,298] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:53,450] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:53,589] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:53,728] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:53,867] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:54,007] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:54,146] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:21:55,002] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:55,148] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:55,292] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:55,437] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:55,584] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:55,731] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:55,877] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:56,021] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:21:56,842] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:21:57,013] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:21:57,174] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:21:57,336] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:21:57,497] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:21:57,659] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:22:03,035] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '我认为跑步最重要的就是给我带来了身体健康。', 'times': [{'w': '我', 'bg': 0.0, 'ed': 0.7000000000000001}, {'w': '认', 'bg': 0.7000000000000001, 'ed': 0.84}, {'w': '为', 'bg': 0.84, 'ed': 1.0}, {'w': '跑', 'bg': 1.0, 'ed': 1.18}, {'w': '步', 'bg': 1.18, 'ed': 1.36}, {'w': '最', 'bg': 1.36, 'ed': 1.5}, {'w': '重', 'bg': 1.5, 'ed': 1.6400000000000001}, {'w': '要', 'bg': 1.6400000000000001, 'ed': 1.78}, {'w': '的', 'bg': 1.78, 'ed': 1.9000000000000001}, {'w': '就', 'bg': 1.9000000000000001, 'ed': 2.06}, {'w': '是', 'bg': 2.06, 'ed': 2.62}, {'w': '给', 'bg': 2.62, 'ed': 3.16}, {'w': '我', 'bg': 3.16, 'ed': 3.3200000000000003}, {'w': '带', 'bg': 3.3200000000000003, 'ed': 3.48}, {'w': '来', 'bg': 3.48, 'ed': 3.62}, {'w': '了', 'bg': 3.62, 'ed': 3.7600000000000002}, {'w': '身', 'bg': 3.7600000000000002, 'ed': 3.9}, {'w': '体', 'bg': 3.9, 'ed': 4.0600000000000005}, {'w': '健', 'bg': 4.0600000000000005, 'ed': 4.26}, {'w': '康', 'bg': 4.26, 'ed': 4.96}]}
  [2022-05-07 11:22:03,035] [    INFO] - audio duration: 4.9968125, elapsed time: 15.974023818969727, RTF=3.1968427510477384
  [2022-05-07 11:22:03,037] [    INFO] - asr websocket client finished
  [2022-05-07 11:22:03,037] [    INFO] - 我认为跑步最重要的就是给我带来了身体健康。
  [2022-05-07 11:22:03,037] [    INFO] - Response time 15.977116 s.
  ```

- 使用脚本调用
  
  若 `127.0.0.1` 不能访问，则需要使用实际服务 IP 地址

  ```bash
  python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --wavfile ./zh.wav
  ```
  输出:
  ```text
  [2022-05-07 11:11:02,984] [    INFO] - Start to do streaming asr client
  [2022-05-07 11:11:02,985] [    INFO] - asr websocket client start
  [2022-05-07 11:11:02,985] [    INFO] - endpoint: ws://127.0.0.1:8490/paddlespeech/asr/streaming
  [2022-05-07 11:11:02,986] [    INFO] - start to process the wavscp: ./zh.wav
  [2022-05-07 11:11:03,006] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
  [2022-05-07 11:11:03,021] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,034] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,046] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,058] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,070] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,081] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,092] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,102] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,629] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,638] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,645] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,653] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,661] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,668] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:03,676] [    INFO] - client receive msg={'result': ''}
  [2022-05-07 11:11:04,402] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:04,510] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:04,619] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:04,743] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:04,849] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:04,956] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:05,063] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:05,170] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-07 11:11:05,876] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:06,019] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:06,184] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:06,342] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:06,537] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:06,727] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:06,871] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-07 11:11:07,617] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:07,769] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:07,905] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:08,043] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:08,186] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:08,326] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:08,466] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:08,611] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-07 11:11:09,431] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:09,571] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:09,714] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:09,853] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:09,992] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:10,129] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:10,266] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-07 11:11:11,113] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:11,296] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:11,439] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:11,582] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:11,727] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:11,869] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:12,011] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:12,153] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-07 11:11:12,969] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:11:13,137] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:11:13,297] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:11:13,456] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:11:13,615] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:11:13,776] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-07 11:11:18,915] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '我认为跑步最重要的就是给我带来了身体健康。', 'times': [{'w': '我', 'bg': 0.0, 'ed': 0.7000000000000001}, {'w': '认', 'bg': 0.7000000000000001, 'ed': 0.84}, {'w': '为', 'bg': 0.84, 'ed': 1.0}, {'w': '跑', 'bg': 1.0, 'ed': 1.18}, {'w': '步', 'bg': 1.18, 'ed': 1.36}, {'w': '最', 'bg': 1.36, 'ed': 1.5}, {'w': '重', 'bg': 1.5, 'ed': 1.6400000000000001}, {'w': '要', 'bg': 1.6400000000000001, 'ed': 1.78}, {'w': '的', 'bg': 1.78, 'ed': 1.9000000000000001}, {'w': '就', 'bg': 1.9000000000000001, 'ed': 2.06}, {'w': '是', 'bg': 2.06, 'ed': 2.62}, {'w': '给', 'bg': 2.62, 'ed': 3.16}, {'w': '我', 'bg': 3.16, 'ed': 3.3200000000000003}, {'w': '带', 'bg': 3.3200000000000003, 'ed': 3.48}, {'w': '来', 'bg': 3.48, 'ed': 3.62}, {'w': '了', 'bg': 3.62, 'ed': 3.7600000000000002}, {'w': '身', 'bg': 3.7600000000000002, 'ed': 3.9}, {'w': '体', 'bg': 3.9, 'ed': 4.0600000000000005}, {'w': '健', 'bg': 4.0600000000000005, 'ed': 4.26}, {'w': '康', 'bg': 4.26, 'ed': 4.96}]}
  [2022-05-07 11:11:18,915] [    INFO] - audio duration: 4.9968125, elapsed time: 15.928460597991943, RTF=3.187724293835709
  [2022-05-07 11:11:18,916] [    INFO] - asr websocket client finished : 我认为跑步最重要的就是给我带来了身体健康
  ```

## 从音频文件(.wav 格式 或者.mp3 格式)生成字幕文件 (.srt 格式)

**注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数将语音识别和标点预测部署在不同的 `gpu` 上。

使用 `streaming_asr_server.py` 和 `punc_server.py` 两个服务，分别启动流式语音识别和标点预测服务。调用 `websocket_client.py` 脚本可以同时调用流式语音识别和标点预测服务，将会生成对应的字幕文件(.srt格式)。

**使用该脚本前需要安装mffpeg**

**应该在对应的`.../demos/streaming_asr_server/`目录下运行以下脚本**

### 1. 启动服务端

```bash
Note: streaming speech recognition and punctuation prediction are configured on different graphics cards through configuration files
paddlespeech_server start --config_file ./conf/ws_conformer_wenetspeech_application.yaml
```

Open another terminal run the following commands:
```bash
paddlespeech_server start --config_file conf/punc_application.yaml
```

### 2. 启动客户端

  ```bash
  python3 local/websocket_client_srt.py --server_ip 127.0.0.1 --port 8090 --punc.server_ip 127.0.0.1 --punc.port 8190 --wavfile ../../data/认知.mp3
  ```
  Output:
  ```text
  [2023-03-30 23:26:13,991] [    INFO] - Start to do streaming asr client
[2023-03-30 23:26:13,994] [    INFO] - asr websocket client start
[2023-03-30 23:26:13,994] [    INFO] - endpoint: http://127.0.0.1:8190/paddlespeech/text
[2023-03-30 23:26:13,994] [    INFO] - endpoint: ws://127.0.0.1:8090/paddlespeech/asr/streaming
[2023-03-30 23:26:14,475] [    INFO] - /home/fxb/PaddleSpeech-develop/data/认知.mp3 converted to /home/fxb/PaddleSpeech-develop/data/认知.wav
[2023-03-30 23:26:14,476] [    INFO] - start to process the wavscp: /home/fxb/PaddleSpeech-develop/data/认知.wav
[2023-03-30 23:26:14,515] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
[2023-03-30 23:26:14,533] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,545] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,556] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,572] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,588] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,600] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,613] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:14,626] [    INFO] - client receive msg={'result': ''}
[2023-03-30 23:26:15,122] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,135] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,154] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,163] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,175] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,185] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,196] [    INFO] - client receive msg={'result': '第一部'}
[2023-03-30 23:26:15,637] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,648] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,657] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,666] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,676] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,683] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,691] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:15,703] [    INFO] - client receive msg={'result': '第一部分是认'}
[2023-03-30 23:26:16,146] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,159] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,167] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,177] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,187] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,197] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,210] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,694] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,704] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,713] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,725] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,737] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,749] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,759] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:16,770] [    INFO] - client receive msg={'result': '第一部分是认知部分'}
[2023-03-30 23:26:17,279] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,302] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,316] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,332] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,343] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,358] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,373] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通'}
[2023-03-30 23:26:17,958] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:17,971] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:17,987] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:18,000] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:18,017] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:18,028] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:18,038] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:18,049] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图'}
[2023-03-30 23:26:18,653] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:18,689] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:18,701] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:18,712] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:18,723] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:18,750] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:18,767] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本'}
[2023-03-30 23:26:19,295] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,307] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,323] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,332] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,342] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,349] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,373] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:19,389] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式'}
[2023-03-30 23:26:20,046] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,055] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,067] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,076] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,094] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,124] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,135] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生'}
[2023-03-30 23:26:20,732] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,742] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,757] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,770] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,782] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,798] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,815] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:20,834] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解'}
[2023-03-30 23:26:21,390] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:21,405] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:21,416] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:21,428] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:21,448] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:21,459] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:21,473] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感'}
[2023-03-30 23:26:22,065] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,085] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,110] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,118] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,137] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,144] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,154] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,169] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作'}
[2023-03-30 23:26:22,698] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:22,709] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:22,731] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:22,743] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:22,755] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:22,771] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:22,782] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理'}
[2023-03-30 23:26:23,415] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,430] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,442] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,456] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,470] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,487] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,498] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:23,524] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生'}
[2023-03-30 23:26:24,200] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,210] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,219] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,231] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,250] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,262] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,272] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备'}
[2023-03-30 23:26:24,898] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:24,903] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:24,907] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:24,932] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:24,957] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:24,979] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:24,991] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:25,011] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致'}
[2023-03-30 23:26:25,616] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,625] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,648] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,658] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,669] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,681] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,690] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:25,707] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,378] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,384] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,389] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,397] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,402] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,415] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:26,428] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知'}
[2023-03-30 23:26:27,008] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,018] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,026] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,037] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,046] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,054] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,062] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,070] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使'}
[2023-03-30 23:26:27,735] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:27,745] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:27,755] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:27,769] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:27,783] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:27,794] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:27,804] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传'}
[2023-03-30 23:26:28,454] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,472] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,481] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,489] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,499] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,533] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,543] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:28,556] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内'}
[2023-03-30 23:26:29,212] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:29,222] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:29,233] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:29,246] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:29,258] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:29,270] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:29,286] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图'}
[2023-03-30 23:26:30,003] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,013] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,038] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,048] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,062] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,074] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,114] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,125] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅'}
[2023-03-30 23:26:30,856] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:30,876] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:30,885] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:30,897] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:30,914] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:30,940] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:30,952] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说'}
[2023-03-30 23:26:31,655] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,696] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,709] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,718] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,727] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,740] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,757] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:31,768] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明'}
[2023-03-30 23:26:32,476] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:32,486] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:32,495] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:32,549] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:32,560] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:32,574] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:32,590] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助'}
[2023-03-30 23:26:33,338] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,356] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,368] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,386] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,397] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,409] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,424] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:33,434] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生'}
[2023-03-30 23:26:34,352] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:34,364] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:34,377] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:34,395] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:34,410] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:34,423] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:34,434] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感'}
[2023-03-30 23:26:35,373] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,397] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,410] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,420] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,437] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,448] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,460] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:35,473] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有'}
[2023-03-30 23:26:36,288] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:36,297] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:36,306] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:36,326] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:36,336] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:36,351] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:36,365] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的'}
[2023-03-30 23:26:37,164] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,173] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,182] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,192] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,204] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,232] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,238] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:37,252] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象'}
[2023-03-30 23:26:38,084] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:38,093] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:38,106] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:38,122] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:38,140] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:38,181] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:38,206] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后'}
[2023-03-30 23:26:39,094] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,111] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,132] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,150] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,174] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,190] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,197] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:39,212] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合'}
[2023-03-30 23:26:40,009] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,094] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,105] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,128] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,149] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,173] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,189] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,200] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实'}
[2023-03-30 23:26:40,952] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:40,973] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:40,986] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:40,999] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:41,013] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:41,022] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:41,033] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用'}
[2023-03-30 23:26:41,819] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,832] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,845] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,878] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,886] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,893] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,925] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:41,935] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升'}
[2023-03-30 23:26:42,562] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:42,589] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:42,621] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:42,634] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:42,644] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:42,657] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:42,668] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对'}
[2023-03-30 23:26:43,380] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,389] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,436] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,448] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,462] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,472] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,486] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:43,496] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴'}
[2023-03-30 23:26:44,346] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:44,356] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:44,364] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:44,374] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:44,389] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:44,398] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:44,420] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以'}
[2023-03-30 23:26:45,226] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以及意义感'}
[2023-03-30 23:26:45,235] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以及意义感'}
[2023-03-30 23:26:45,258] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以及意义感'}
[2023-03-30 23:26:45,273] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以及意义感'}
[2023-03-30 23:26:45,295] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以及意义感'}
[2023-03-30 23:26:45,306] [    INFO] - client receive msg={'result': '第一部分是认知部分该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理让学生对设备有大致的认知随后使用真实传感器的内部构造图辅以文字说明进一步帮助学生对传感器有更深刻的印象最后结合具体的实践应用提升学生对实训的兴趣以及意义感'}
[2023-03-30 23:26:46,380] [    INFO] - client punctuation restored msg={'result': '第一部分是认知部分，该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理，让学生对设备有大致的认知。随后使用真实传感器的内部构造图，辅以文字说明，进一步帮助学生对传感器有更深刻的印象，最后结合具体的实践应用，提升学生对实训的兴趣以及意义感。'}
[2023-03-30 23:27:01,059] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '第一部分是认知部分，该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理，让学生对设备有大致的认知。随后使用真实传感器的内部构造图，辅以文字说明，进一步帮助学生对传感器有更深刻的印象，最后结合具体的实践应用，提升学生对实训的兴趣以及意义感。', 'times': [{'w': '第', 'bg': 0.0, 'ed': 0.36}, {'w': '一', 'bg': 0.36, 'ed': 0.48}, {'w': '部', 'bg': 0.48, 'ed': 0.62}, {'w': '分', 'bg': 0.62, 'ed': 0.8200000000000001}, {'w': '是', 'bg': 0.8200000000000001, 'ed': 1.08}, {'w': '认', 'bg': 1.08, 'ed': 1.28}, {'w': '知', 'bg': 1.28, 'ed': 1.44}, {'w': '部', 'bg': 1.44, 'ed': 1.58}, {'w': '分', 'bg': 1.58, 'ed': 2.1}, {'w': '该', 'bg': 2.1, 'ed': 2.6}, {'w': '部', 'bg': 2.6, 'ed': 2.72}, {'w': '分', 'bg': 2.72, 'ed': 2.94}, {'w': '通', 'bg': 2.94, 'ed': 3.16}, {'w': '过', 'bg': 3.16, 'ed': 3.36}, {'w': '示', 'bg': 3.36, 'ed': 3.54}, {'w': '意', 'bg': 3.54, 'ed': 3.68}, {'w': '图', 'bg': 3.68, 'ed': 3.9}, {'w': '和', 'bg': 3.9, 'ed': 4.14}, {'w': '文', 'bg': 4.14, 'ed': 4.32}, {'w': '本', 'bg': 4.32, 'ed': 4.46}, {'w': '的', 'bg': 4.46, 'ed': 4.58}, {'w': '形', 'bg': 4.58, 'ed': 4.72}, {'w': '式', 'bg': 4.72, 'ed': 5.0}, {'w': '向', 'bg': 5.0, 'ed': 5.32}, {'w': '学', 'bg': 5.32, 'ed': 5.5}, {'w': '生', 'bg': 5.5, 'ed': 5.66}, {'w': '讲', 'bg': 5.66, 'ed': 5.86}, {'w': '解', 'bg': 5.86, 'ed': 6.18}, {'w': '主', 'bg': 6.18, 'ed': 6.46}, {'w': '要', 'bg': 6.46, 'ed': 6.62}, {'w': '传', 'bg': 6.62, 'ed': 6.8}, {'w': '感', 'bg': 6.8, 'ed': 7.0}, {'w': '器', 'bg': 7.0, 'ed': 7.16}, {'w': '的', 'bg': 7.16, 'ed': 7.28}, {'w': '工', 'bg': 7.28, 'ed': 7.44}, {'w': '作', 'bg': 7.44, 'ed': 7.6000000000000005}, {'w': '原', 'bg': 7.6000000000000005, 'ed': 7.74}, {'w': '理', 'bg': 7.74, 'ed': 8.06}, {'w': '让', 'bg': 8.06, 'ed': 8.44}, {'w': '学', 'bg': 8.44, 'ed': 8.64}, {'w': '生', 'bg': 8.64, 'ed': 8.84}, {'w': '对', 'bg': 8.84, 'ed': 9.06}, {'w': '设', 'bg': 9.06, 'ed': 9.24}, {'w': '备', 'bg': 9.24, 'ed': 9.52}, {'w': '有', 'bg': 9.52, 'ed': 9.86}, {'w': '大', 'bg': 9.86, 'ed': 10.1}, {'w': '致', 'bg': 10.1, 'ed': 10.24}, {'w': '的', 'bg': 10.24, 'ed': 10.36}, {'w': '认', 'bg': 10.36, 'ed': 10.5}, {'w': '知', 'bg': 10.5, 'ed': 11.040000000000001}, {'w': '随', 'bg': 11.040000000000001, 'ed': 11.56}, {'w': '后', 'bg': 11.56, 'ed': 11.82}, {'w': '使', 'bg': 11.82, 'ed': 12.1}, {'w': '用', 'bg': 12.1, 'ed': 12.26}, {'w': '真', 'bg': 12.26, 'ed': 12.44}, {'w': '实', 'bg': 12.44, 'ed': 12.620000000000001}, {'w': '传', 'bg': 12.620000000000001, 'ed': 12.780000000000001}, {'w': '感', 'bg': 12.780000000000001, 'ed': 12.94}, {'w': '器', 'bg': 12.94, 'ed': 13.1}, {'w': '的', 'bg': 13.1, 'ed': 13.26}, {'w': '内', 'bg': 13.26, 'ed': 13.42}, {'w': '部', 'bg': 13.42, 'ed': 13.56}, {'w': '构', 'bg': 13.56, 'ed': 13.700000000000001}, {'w': '造', 'bg': 13.700000000000001, 'ed': 13.86}, {'w': '图', 'bg': 13.86, 'ed': 14.280000000000001}, {'w': '辅', 'bg': 14.280000000000001, 'ed': 14.66}, {'w': '以', 'bg': 14.66, 'ed': 14.82}, {'w': '文', 'bg': 14.82, 'ed': 15.0}, {'w': '字', 'bg': 15.0, 'ed': 15.16}, {'w': '说', 'bg': 15.16, 'ed': 15.32}, {'w': '明', 'bg': 15.32, 'ed': 15.72}, {'w': '进', 'bg': 15.72, 'ed': 16.1}, {'w': '一', 'bg': 16.1, 'ed': 16.2}, {'w': '步', 'bg': 16.2, 'ed': 16.32}, {'w': '帮', 'bg': 16.32, 'ed': 16.48}, {'w': '助', 'bg': 16.48, 'ed': 16.66}, {'w': '学', 'bg': 16.66, 'ed': 16.82}, {'w': '生', 'bg': 16.82, 'ed': 17.12}, {'w': '对', 'bg': 17.12, 'ed': 17.48}, {'w': '传', 'bg': 17.48, 'ed': 17.66}, {'w': '感', 'bg': 17.66, 'ed': 17.84}, {'w': '器', 'bg': 17.84, 'ed': 18.12}, {'w': '有', 'bg': 18.12, 'ed': 18.42}, {'w': '更', 'bg': 18.42, 'ed': 18.66}, {'w': '深', 'bg': 18.66, 'ed': 18.88}, {'w': '刻', 'bg': 18.88, 'ed': 19.04}, {'w': '的', 'bg': 19.04, 'ed': 19.16}, {'w': '印', 'bg': 19.16, 'ed': 19.3}, {'w': '象', 'bg': 19.3, 'ed': 19.8}, {'w': '最', 'bg': 19.8, 'ed': 20.3}, {'w': '后', 'bg': 20.3, 'ed': 20.62}, {'w': '结', 'bg': 20.62, 'ed': 20.96}, {'w': '合', 'bg': 20.96, 'ed': 21.14}, {'w': '具', 'bg': 21.14, 'ed': 21.3}, {'w': '体', 'bg': 21.3, 'ed': 21.42}, {'w': '的', 'bg': 21.42, 'ed': 21.580000000000002}, {'w': '实', 'bg': 21.580000000000002, 'ed': 21.76}, {'w': '践', 'bg': 21.76, 'ed': 21.92}, {'w': '应', 'bg': 21.92, 'ed': 22.080000000000002}, {'w': '用', 'bg': 22.080000000000002, 'ed': 22.44}, {'w': '提', 'bg': 22.44, 'ed': 22.78}, {'w': '升', 'bg': 22.78, 'ed': 22.94}, {'w': '学', 'bg': 22.94, 'ed': 23.12}, {'w': '生', 'bg': 23.12, 'ed': 23.34}, {'w': '对', 'bg': 23.34, 'ed': 23.62}, {'w': '实', 'bg': 23.62, 'ed': 23.82}, {'w': '训', 'bg': 23.82, 'ed': 23.96}, {'w': '的', 'bg': 23.96, 'ed': 24.12}, {'w': '兴', 'bg': 24.12, 'ed': 24.3}, {'w': '趣', 'bg': 24.3, 'ed': 24.6}, {'w': '以', 'bg': 24.6, 'ed': 24.88}, {'w': '及', 'bg': 24.88, 'ed': 25.12}, {'w': '意', 'bg': 25.12, 'ed': 25.34}, {'w': '义', 'bg': 25.34, 'ed': 25.46}, {'w': '感', 'bg': 25.46, 'ed': 26.04}]}
[2023-03-30 23:27:01,060] [    INFO] - audio duration: 26.04, elapsed time: 46.581613540649414, RTF=1.7888484462614982
sentences:  ['第一部分是认知部分', '该部分通过示意图和文本的形式向学生讲解主要传感器的工作原理', '让学生对设备有大致的认知', '随后使用真实传感器的内部构造图', '辅以文字说明', '进一步帮助学生对传感器有更深刻的印象', '最后结合具体的实践应用', '提升学生对实训的兴趣以及意义感']
relative_times:  [[0.0, 2.1], [2.1, 8.06], [8.06, 11.040000000000001], [11.040000000000001, 14.280000000000001], [14.280000000000001, 15.72], [15.72, 19.8], [19.8, 22.44], [22.44, 26.04]]
[2023-03-30 23:27:01,076] [    INFO] - results saved to /home/fxb/PaddleSpeech-develop/data/认知.srt
  ```
