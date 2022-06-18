(简体中文|[English](./README.md))

# 语音内容搜索
## 介绍
语音内容搜索是一项用计算机程序获取转录语音内容关键词时间戳的技术。

这个 demo 是一个从给定音频文件获取其文本中关键词时间戳的实现，它可以通过使用 `PaddleSpeech` 的单个命令或 python 中的几行代码来实现。

当前示例中检索词是
```
我
康
```
## 使用方法
### 1. 安装
请看[安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md)。

你可以从 medium，hard 三中方式中选择一种方式安装。
依赖参见 requirements.txt, 安装依赖

```
pip install -r requriement.txt 
```

### 2. 准备输入
这个 demo 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 demo 的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```
### 3. 使用方法
- 命令行 (推荐使用)
  ```bash
  # 中文
  paddlespeech_client acs --server_ip 127.0.0.1 --port 8090 --input ./zh.wav 
  ```
  
  使用方法：
  ```bash
  paddlespeech acs --help
  ```
  参数：
  - `input`(必须输入)：用于识别的音频文件。
  - `server_ip`: 服务的ip。
  - `port`：服务的端口。
  - `lang`：模型语言，默认值：`zh`。
  - `sample_rate`：音频采样率，默认值：`16000`。
  - `audio_format`: 音频的格式。

  输出：
  ```bash
  [2022-05-15 15:00:58,185] [    INFO] - acs http client start
  [2022-05-15 15:00:58,185] [    INFO] - endpoint: http://127.0.0.1:8490/paddlespeech/asr/search
  [2022-05-15 15:01:03,220] [    INFO] - acs http client finished
  [2022-05-15 15:01:03,221] [    INFO] - ACS result: {'transcription': '我认为跑步最重要的就是给我带来了身体健康', 'acs': [{'w': '我', 'bg': 0, 'ed': 1.6800000000000002}, {'w': '我', 'bg': 2.1, 'ed': 4.28}, {'w': '康', 'bg': 3.2, 'ed': 4.92}]}
  [2022-05-15 15:01:03,221] [    INFO] - Response time 5.036084 s.
  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import ACSClientExecutor

  acs_executor = ACSClientExecutor()
  res = acs_executor(
      input='./zh.wav',
      server_ip="127.0.0.1",
      port=8490,)
  print(res)
  ```

  输出：
  ```bash
  [2022-05-15 15:08:13,955] [    INFO] - acs http client start
  [2022-05-15 15:08:13,956] [    INFO] - endpoint: http://127.0.0.1:8490/paddlespeech/asr/search
  [2022-05-15 15:08:19,026] [    INFO] - acs http client finished
  {'transcription': '我认为跑步最重要的就是给我带来了身体健康', 'acs': [{'w': '我', 'bg': 0, 'ed': 1.6800000000000002}, {'w': '我', 'bg': 2.1, 'ed': 4.28}, {'w': '康', 'bg': 3.2, 'ed': 4.92}]}
  ```
