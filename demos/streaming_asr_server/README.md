([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the streaming speech service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.


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
  # start the service
   paddlespeech_server start --config_file ./conf/ws_conformer_application.yaml
  ```

  Usage:
  
  ```bash
  paddlespeech_server start --help
  ```
  Arguments:
  - `config_file`: yaml file of the app, defalut: ./conf/ws_conformer_application.yaml
  - `log_file`: log file. Default: ./log/paddlespeech.log

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
