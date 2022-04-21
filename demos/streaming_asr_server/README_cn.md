([English](./README.md)|中文)

# 语音服务

## 介绍
这个demo是一个启动流式语音服务和访问服务的实现。 它可以通过使用`paddlespeech_server` 和 `paddlespeech_client`的单个命令或 python 的几行代码来实现。


## 使用方法
### 1. 安装
请看 [安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

推荐使用 **paddlepaddle 2.2.1** 或以上版本。
你可以从 medium，hard 三中方式中选择一种方式安装 PaddleSpeech。


### 2. 准备配置文件
配置文件可参见 `conf/ws_application.yaml` 和 `conf/ws_conformer_application.yaml` 。
目前服务集成的模型有： DeepSpeech2和conformer模型。


这个 ASR client 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 ASR client的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 3. 服务端使用方法
- 命令行 (推荐使用)

  ```bash
  # 启动服务
  paddlespeech_server start --config_file ./conf/ws_conformer_application.yaml
  ```

  使用方法：
  
  ```bash
  paddlespeech_server start --help
  ```
  参数:
  - `config_file`: 服务的配置文件，默认： ./conf/ws_conformer_application.yaml
  - `log_file`: log 文件. 默认：./log/paddlespeech.log

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
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
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
  ```python
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
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
    set kaiming_uniform
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

    输出:

    ```bash
        [2022-04-21 15:59:03,904] [    INFO] - receive msg={"status": "ok", "signal": "server_ready"}
        [2022-04-21 15:59:03,960] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:03,973] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:03,987] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,000] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,012] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,024] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,036] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,047] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,607] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,620] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,633] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,645] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,657] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,669] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,680] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:05,176] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,185] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,192] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,200] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,208] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,216] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,224] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,232] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,724] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,732] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,740] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,747] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,755] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,763] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,770] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:06,271] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,279] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,287] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,294] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,302] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,310] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,318] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,326] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,833] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,842] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,850] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,858] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,866] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,874] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,882] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:07,400] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,408] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,416] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,424] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,432] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,440] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,447] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,455] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,984] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:07,992] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,001] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,008] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,016] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,024] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:12,883] [    INFO] - final receive msg={'status': 'ok', 'signal': 'finished', 'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:12,884] [    INFO] - 我认为跑步最重要的就是给我带来了身体健康
        [2022-04-21 15:59:12,884] [    INFO] - Response time 9.051567 s.
    ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import ASROnlineClientExecutor
  import json

  asrclient_executor = ASROnlineClientExecutor()
  res = asrclient_executor(
      input="./zh.wav",
      server_ip="127.0.0.1",
      port=8090,
      sample_rate=16000,
      lang="zh_cn",
      audio_format="wav")
  print(res.json())
  ```

  输出:
  ```bash
        [2022-04-21 15:59:03,904] [    INFO] - receive msg={"status": "ok", "signal": "server_ready"}
        [2022-04-21 15:59:03,960] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:03,973] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:03,987] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,000] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,012] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,024] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,036] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,047] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,607] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,620] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,633] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,645] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,657] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,669] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:04,680] [    INFO] - receive msg={'asr_results': ''}
        [2022-04-21 15:59:05,176] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,185] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,192] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,200] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,208] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,216] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,224] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,232] [    INFO] - receive msg={'asr_results': '我认为跑'}
        [2022-04-21 15:59:05,724] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,732] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,740] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,747] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,755] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,763] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:05,770] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的'}
        [2022-04-21 15:59:06,271] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,279] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,287] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,294] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,302] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,310] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,318] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,326] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是'}
        [2022-04-21 15:59:06,833] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,842] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,850] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,858] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,866] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,874] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:06,882] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给'}
        [2022-04-21 15:59:07,400] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,408] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,416] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,424] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,432] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,440] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,447] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,455] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了'}
        [2022-04-21 15:59:07,984] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:07,992] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,001] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,008] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,016] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:08,024] [    INFO] - receive msg={'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:12,883] [    INFO] - final receive msg={'status': 'ok', 'signal': 'finished', 'asr_results': '我认为跑步最重要的就是给我带来了身体健康'}
        [2022-04-21 15:59:12,884] [    INFO] - 我认为跑步最重要的就是给我带来了身体健康
  ```