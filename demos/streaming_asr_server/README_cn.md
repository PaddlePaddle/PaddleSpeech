([English](./README.md)|中文)

# 语音服务

## 介绍
这个demo是一个启动流式语音服务和访问服务的实现。 它可以通过使用`paddlespeech_server` 和 `paddlespeech_client`的单个命令或 python 的几行代码来实现。

**流式语音识别服务只支持 `weboscket` 协议，不支持 `http` 协议。**

## 使用方法
### 1. 安装
安装 PaddleSpeech 的详细过程请看 [安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md)。

推荐使用 **paddlepaddle 2.2.1** 或以上版本。
你可以从medium，hard 两种方式中选择一种方式安装 PaddleSpeech。


### 2. 准备配置文件

流式ASR的服务启动脚本和服务测试脚本存放在 `PaddleSpeech/demos/streaming_asr_server` 目录。
下载好 `PaddleSpeech` 之后，进入到 `PaddleSpeech/demos/streaming_asr_server` 目录。
配置文件可参见该目录下 `conf/ws_application.yaml` 和 `conf/ws_conformer_application.yaml` 。

目前服务集成的模型有： DeepSpeech2和 conformer模型，对应的配置文件如下：
* DeepSpeech: `conf/ws_application.yaml`
* conformer: `conf/ws_conformer_application.yaml`



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
  paddlespeech_server start --config_file ./conf/ws_conformer_application.yaml
  ```

  使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  参数:
  - `config_file`: 服务的配置文件，默认： `./conf/application.yaml`
  - `log_file`: log 文件. 默认：`./log/paddlespeech.log`

  输出:
  ```bash
    [2022-04-21 15:52:18,126] [    INFO] - create the online asr engine instance
    [2022-04-21 15:52:18,127] [    INFO] - paddlespeech_server set the device: cpu
    [2022-04-21 15:52:18,128] [    INFO] - Load the pretrained model, tag = conformer_online_multicn-zh-16k
    [2022-04-21 15:52:18,128] [    INFO] - File /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/asr1_chunk_conformer_multi_cn_ckpt_0.2.3.model.tar.gz md5 checking...
    [2022-04-21 15:52:18,727] [    INFO] - Use pretrained model stored in: /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/model.yaml
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/exp/chunk_conformer/checkpoints/multi_cn.pdparams
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/exp/chunk_conformer/checkpoints/multi_cn.pdparams
    [2022-04-21 15:52:19,446] [    INFO] - start to create the stream conformer asr engine
    [2022-04-21 15:52:19,473] [    INFO] - model name: conformer_online
    [2022-04-21 15:52:21,731] [    INFO] - create the transformer like model success
    [2022-04-21 15:52:21,733] [    INFO] - Initialize ASR server engine successfully.
    INFO:     Started server process [11173]
    [2022-04-21 15:52:21] [INFO] [server.py:75] Started server process [11173]
    INFO:     Waiting for application startup.
    [2022-04-21 15:52:21] [INFO] [on.py:45] Waiting for application startup.
    INFO:     Application startup complete.
    [2022-04-21 15:52:21] [INFO] [on.py:59] Application startup complete.
    /home/users/xiongxinlei/.conda/envs/paddlespeech/lib/python3.9/asyncio/base_events.py:1460: DeprecationWarning: The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10.
    infos = await tasks.gather(*fs, loop=self)
    /home/users/xiongxinlei/.conda/envs/paddlespeech/lib/python3.9/asyncio/base_events.py:1518: DeprecationWarning: The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10.
    await tasks.sleep(0, loop=self)
    INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
    [2022-04-21 15:52:21] [INFO] [server.py:206] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  ```

- Python API
  **注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数部署在 `gpu` 上。
  ```python
  # 在 PaddleSpeech/demos/streaming_asr_server 目录
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/ws_conformer_application.yaml", 
      log_file="./log/paddlespeech.log")
  ```

  输出：
  ```bash
    [2022-04-21 15:52:18,126] [    INFO] - create the online asr engine instance
    [2022-04-21 15:52:18,127] [    INFO] - paddlespeech_server set the device: cpu
    [2022-04-21 15:52:18,128] [    INFO] - Load the pretrained model, tag = conformer_online_multicn-zh-16k
    [2022-04-21 15:52:18,128] [    INFO] - File /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/asr1_chunk_conformer_multi_cn_ckpt_0.2.3.model.tar.gz md5 checking...
    [2022-04-21 15:52:18,727] [    INFO] - Use pretrained model stored in: /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/model.yaml
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/exp/chunk_conformer/checkpoints/multi_cn.pdparams
    [2022-04-21 15:52:18,727] [    INFO] - /home/users/xiongxinlei/.paddlespeech/models/conformer_online_multicn-zh-16k/exp/chunk_conformer/checkpoints/multi_cn.pdparams
    [2022-04-21 15:52:19,446] [    INFO] - start to create the stream conformer asr engine
    [2022-04-21 15:52:19,473] [    INFO] - model name: conformer_online
    [2022-04-21 15:52:21,731] [    INFO] - create the transformer like model success
    [2022-04-21 15:52:21,733] [    INFO] - Initialize ASR server engine successfully.
    INFO:     Started server process [11173]
    [2022-04-21 15:52:21] [INFO] [server.py:75] Started server process [11173]
    INFO:     Waiting for application startup.
    [2022-04-21 15:52:21] [INFO] [on.py:45] Waiting for application startup.
    INFO:     Application startup complete.
    [2022-04-21 15:52:21] [INFO] [on.py:59] Application startup complete.
    /home/users/xiongxinlei/.conda/envs/paddlespeech/lib/python3.9/asyncio/base_events.py:1460: DeprecationWarning: The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10.
    infos = await tasks.gather(*fs, loop=self)
    /home/users/xiongxinlei/.conda/envs/paddlespeech/lib/python3.9/asyncio/base_events.py:1518: DeprecationWarning: The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10.
    await tasks.sleep(0, loop=self)
    INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
    [2022-04-21 15:52:21] [INFO] [server.py:206] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  ```

### 4. ASR 客户端使用方法

**注意：** 初次使用客户端时响应时间会略长
- 命令行 (推荐使用)
   ```
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

    ```bash
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
  ```bash
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
  ``` bash
  在 PaddleSpeech/demos/streaming_asr_server 目录下启动标点预测服务
  paddlespeech_server start --config_file conf/punc_application.yaml
  ```


   使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  
  参数：
  - `config_file`: 服务的配置文件。
  - `log_file`: log 文件。


  输出：
  ``` bash
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

   输出
   ```
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
   ```
   paddlespeech_client text --server_ip 127.0.0.1 --port 8190 --input "我认为跑步最重要的就是给我带来了身体健康"
   ```
  
  输出
  ```
  [2022-05-02 18:12:29,767] [    INFO] - The punc text: 我认为跑步最重要的就是给我带来了身体健康。
  [2022-05-02 18:12:29,767] [    INFO] - Response time 0.096548 s.
  ```

- Python3 API

  ```python
  from paddlespeech.server.bin.paddlespeech_client import TextClientExecutor

  textclient_executor = TextClientExecutor()
  res = textclient_executor(
      input="我认为跑步最重要的就是给我带来了身体健康",
      server_ip="127.0.0.1",
      port=8190,)
  print(res)
  ```

  输出：
  ``` bash
  我认为跑步最重要的就是给我带来了身体健康。
  ```


## 联合流式语音识别和标点预测
**注意:** 默认部署在 `cpu` 设备上，可以通过修改服务配置文件中 `device` 参数将语音识别和标点预测部署在不同的 `gpu` 上。

使用 `streaming_asr_server.py` 和 `punc_server.py` 两个服务，分别启动流式语音识别和标点预测服务。调用 `websocket_client.py` 脚本可以同时调用流式语音识别和标点预测服务。

### 1. 启动服务

``` bash
注意：流式语音识别和标点预测通过配置文件配置到不同的显卡上
bash server.sh
```

### 2. 调用服务
- 使用命令行：
  ```
  paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --input ./zh.wav
  ```
  输出：
  ```
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
  ```
  python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --wavfile ./zh.wav
  ```
  输出：
  ```
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

  