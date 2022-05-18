([简体中文](./README_cn.md)|English)
# ACS (Audio Content Search)

## Introduction
ACS, or Audio Content Search, refers to the problem of getting the key word time stamp from automatically transcribe spoken language (speech-to-text). 

This demo is an implementation of obtaining the keyword timestamp in the text from a given audio file. It can be done by a single command or a few lines in python using `PaddleSpeech`. 
Now, the search word in demo is:
```
我
康
```
## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

You can choose one way from meduim and hard to install paddlespeech.

The dependency refers to the requirements.txt
### 2. Prepare Input File
The input of this demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for this demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 3. Usage
- Command Line(Recommended)
  ```bash
  # Chinese
  paddlespeech_client acs --server_ip 127.0.0.1 --port 8090 --input ./zh.wav 
  ```
  
  Usage:
  ```bash
  paddlespeech asr --help
  ```
  Arguments:
  - `input`(required): Audio file to recognize.
  - `server_ip`: the server ip.
  - `port`: the server port.
  - `lang`: the language type of the model. Default: `zh`.
  - `sample_rate`: Sample rate of the model. Default: `16000`.
  - `audio_format`: The audio format.

  Output:
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

  Output:
  ```bash
  [2022-05-15 15:08:13,955] [    INFO] - acs http client start
  [2022-05-15 15:08:13,956] [    INFO] - endpoint: http://127.0.0.1:8490/paddlespeech/asr/search
  [2022-05-15 15:08:19,026] [    INFO] - acs http client finished
  {'transcription': '我认为跑步最重要的就是给我带来了身体健康', 'acs': [{'w': '我', 'bg': 0, 'ed': 1.6800000000000002}, {'w': '我', 'bg': 2.1, 'ed': 4.28}, {'w': '康', 'bg': 3.2, 'ed': 4.92}]}
  ```
