([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the streaming speech service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.

Streaming ASR server only support `websocket` protocol, and doesn't support `http` protocol.

## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.2.1** or above.
You can choose one way from meduim and hard to install paddlespeech.

### 2. Prepare config File
The configuration file can be found in `conf/ws_application.yaml` 和 `conf/ws_conformer_application.yaml`.

At present, the speech tasks integrated by the model include: DeepSpeech2 and conformer.


The input of  ASR client demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for thisASR client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 3. Server Usage
- Command Line (Recommended)

  ```bash
  # in PaddleSpeech/demos/streaming_asr_server start the service
   paddlespeech_server start --config_file ./conf/ws_conformer_application.yaml
  ```

  Usage:
  
  ```bash
  paddlespeech_server start --help
  ```
  Arguments:
  - `config_file`: yaml file of the app, defalut: `./conf/application.yaml`
  - `log_file`: log file. Default: `./log/paddlespeech.log`

  Output:
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
  # in PaddleSpeech/demos/streaming_asr_server directory
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/ws_conformer_application.yaml",
      log_file="./log/paddlespeech.log")
  ```

  Output:
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


### 4. ASR Client Usage

**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)
   ```
   paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --input ./zh.wav
   ```

  Usage:
  
  ```bash
  paddlespeech_client asr_online --help
  ```
  Arguments:
  - `server_ip`: server ip. Default: 127.0.0.1
  - `port`: server port. Default: 8090
  - `input`(required): Audio file to be recognized.
  - `sample_rate`: Audio ampling rate, default: 16000.
  - `lang`: Language. Default: "zh_cn".
  - `audio_format`: Audio format. Default: "wav".
  - `punc.server_ip`: punctuation server ip. Default: None.
  - `punc.server_port`: punctuation server port. Default: None.

  Output:
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

  Output:
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
  ```


## Punctuation service

### 1. Server usage

- Command Line
  ``` bash
  In PaddleSpeech/demos/streaming_asr_server directory to lanuch punctuation service
  paddlespeech_server start --config_file conf/punc_application.yaml
  ```


   Usage:
  ```bash
  paddlespeech_server start --help
  ```
  
  Arguments:
  - `config_file`: configuration file.
  - `log_file`: log file.


  Output:
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

  ```python
  # 在 PaddleSpeech/demos/streaming_asr_server 目录
  from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

  server_executor = ServerExecutor()
  server_executor(
      config_file="./conf/punc_application.yaml", 
      log_file="./log/paddlespeech.log")
  ```

   Output:
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

### 2. Client usage
**Note** The response time will be slightly longer when using the client for the first time

- Command line
   ```
   paddlespeech_client text --server_ip 127.0.0.1 --port 8190 --input "我认为跑步最重要的就是给我带来了身体健康"
   ```
  
  Output
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

  Output:
  ``` bash
  我认为跑步最重要的就是给我带来了身体健康。
  ```


## Join streaming asr and punctuation server
We use `streaming_ asr_server.py` and `punc_server.py` two services to lanuch streaming speech recognition and punctuation prediction services respectively. And the `websocket_client.py` script can be used to call streaming speech recognition and punctuation prediction services at the same time.

### 1. Start two server

``` bash
Note: streaming speech recognition and punctuation prediction are configured on different graphics cards through configuration files
bash server.sh
```

### 2. Call client
- Command line
  ```
  paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --input ./zh.wav
  ```
  Output:
  ```
  [2022-05-02 18:57:46,961] [    INFO] - asr websocket client start
  [2022-05-02 18:57:46,961] [    INFO] - endpoint: ws://127.0.0.1:8290/paddlespeech/asr/streaming
  [2022-05-02 18:57:46,982] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
  [2022-05-02 18:57:46,999] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,011] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,023] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,035] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,046] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,057] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,068] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,079] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,222] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,230] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,239] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,247] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,255] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,263] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,271] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:57:47,462] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,525] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,589] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,649] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,708] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,766] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,824] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:47,881] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:57:48,130] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,200] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,265] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,327] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,389] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,448] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,505] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:57:48,754] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:48,821] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:48,881] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:48,939] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:49,011] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:49,080] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:49,146] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:49,210] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:57:49,452] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:49,516] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:49,581] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:49,645] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:49,706] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:49,763] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:49,818] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:57:50,064] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,125] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,186] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,245] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,301] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,358] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,414] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,469] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:57:50,712] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:50,776] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:50,837] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:50,897] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:50,956] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:51,012] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:51,276] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:57:51,277] [    INFO] - asr websocket client finished
  [2022-05-02 18:57:51,277] [    INFO] - 我认为跑步最重要的就是给我带来了身体健康。
  [2022-05-02 18:57:51,277] [    INFO] - Response time 4.316903 s.
  ```

- Use script
  ```
  python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --wavfile ./zh.wav
  ```
  Output:
  ```
  [2022-05-02 18:29:22,039] [    INFO] - Start to do streaming asr client
  [2022-05-02 18:29:22,040] [    INFO] - asr websocket client start
  [2022-05-02 18:29:22,040] [    INFO] - endpoint: ws://127.0.0.1:8290/paddlespeech/asr/streaming
  [2022-05-02 18:29:22,041] [    INFO] - start to process the wavscp: ./zh.wav
  [2022-05-02 18:29:22,122] [    INFO] - client receive msg={"status": "ok", "signal": "server_ready"}
  [2022-05-02 18:29:22,351] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,360] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,368] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,376] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,384] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,392] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,400] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,408] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,549] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,558] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,567] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,575] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,583] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,591] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,599] [    INFO] - client receive msg={'result': ''}
  [2022-05-02 18:29:22,822] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:22,879] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:22,937] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:22,995] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:23,052] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:23,107] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:23,161] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:23,213] [    INFO] - client receive msg={'result': '我认为，跑'}
  [2022-05-02 18:29:23,454] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:23,515] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:23,575] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:23,630] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:23,684] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:23,736] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:23,789] [    INFO] - client receive msg={'result': '我认为，跑步最重要的。'}
  [2022-05-02 18:29:24,030] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,095] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,156] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,213] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,268] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,323] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,377] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,429] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是。'}
  [2022-05-02 18:29:24,671] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:24,736] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:24,797] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:24,857] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:24,918] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:24,975] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:25,029] [    INFO] - client receive msg={'result': '我认为，跑步最重要的就是给。'}
  [2022-05-02 18:29:25,271] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,336] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,398] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,458] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,521] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,579] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,652] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,722] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了。'}
  [2022-05-02 18:29:25,969] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,034] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,095] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,163] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,229] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,294] [    INFO] - client receive msg={'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,565] [    INFO] - client final receive msg={'status': 'ok', 'signal': 'finished', 'result': '我认为跑步最重要的就是给我带来了身体健康。'}
  [2022-05-02 18:29:26,566] [    INFO] - asr websocket client finished : 我认为跑步最重要的就是给我带来了身体健康。
  ```

  