([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the voice service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.

For service interface definition, please check:
- [PaddleSpeech Server RESTful API](https://github.com/PaddlePaddle/PaddleSpeech/wiki/PaddleSpeech-Server-RESTful-API)


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.4rc** or above.

You can choose one way from easy, meduim and hard to install paddlespeech.

**If you install in easy mode, you need to prepare the yaml file by yourself, you can refer to the yaml file in the conf directory.**

### 2. Prepare config File
The configuration file can be found in `conf/application.yaml` .
Among them, `engine_list` indicates the speech engine that will be included in the service to be started, in the format of `<speech task>_<engine type>`.
At present, the speech tasks integrated by the service include: asr (speech recognition), tts (text to sppech) and cls (audio classification).
Currently the engine type supports two forms: python and inference (Paddle Inference)
**Note:** If the service can be started normally in the container, but the client access IP is unreachable, you can try to replace the `host` address in the configuration file with the local IP address.

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
  ```text
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

  Output:
  ```text
  INFO:     Started server process [529]
  [2022-02-23 14:57:56] [INFO] [server.py:64] Started server process [529]
  INFO:     Waiting for application startup.
  [2022-02-23 14:57:56] [INFO] [on.py:26] Waiting for application startup.
  INFO:     Application startup complete.
  [2022-02-23 14:57:56] [INFO] [on.py:38] Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  [2022-02-23 14:57:56] [INFO] [server.py:204] Uvicorn running on http://0.0.0.0:8090 (Press CTRL+C to quit)
  ```


### 4. ASR Client Usage

The input of  ASR client demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for this ASR client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)

   If `127.0.0.1` is not accessible, you need to use the actual service IP address.

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
  ```text
  [2022-08-01 07:54:01,646] [    INFO] - ASR result: 我认为跑步最重要的就是给我带来了身体健康
  [2022-08-01 07:54:01,646] [    INFO] - Response time 4.898965 s.
  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor

  asrclient_executor = ASRClientExecutor()
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
  ```text
  我认为跑步最重要的就是给我带来了身体健康
  ```
 
### 5. TTS Client Usage
**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)

   If `127.0.0.1` is not accessible, you need to use the actual service IP address

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
    ```text
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
  ```text
  {'description': 'success.'}
  Save synthesized audio successfully on ./output.wav.
  Audio duration: 3.612500 s.
  ```

### 6. CLS Client Usage

Here are sample files for this CLS Client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav 
```

**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)

   If `127.0.0.1` is not accessible, you need to use the actual service IP address.

   ```bash
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
  ```text
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
  ```text
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'topk': 1, 'results': [{'class_name': 'Speech', 'prob': 0.9027184844017029}]}}
  ```


### 7. Speaker Verification Client Usage

Here are sample files for this Speaker Verification Client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav
wget -c https://paddlespeech.bj.bcebos.com/vector/audio/123456789.wav
```

#### 7.1 Extract speaker embedding
**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)

  If `127.0.0.1` is not accessible, you need to use the actual service IP address.

  ``` bash
  paddlespeech_client vector --task spk  --server_ip 127.0.0.1 --port 8090 --input 85236145389.wav
  ```

  Usage:

  ``` bash
  paddlespeech_client vector --help
  ```

  Arguments:
    * server_ip: server ip. Default: 127.0.0.1
    * port: server port. Default: 8090
    * input(required): Input text to generate.
    * task: the task of vector, can be use 'spk' or 'score。Default is 'spk'。
    * enroll: enroll audio
    * test: test audio

  Output:

  ```text
  [2022-08-01 09:01:22,151] [    INFO] - vector http client start
  [2022-08-01 09:01:22,152] [    INFO] - the input audio: 85236145389.wav
  [2022-08-01 09:01:22,152] [    INFO] - endpoint: http://127.0.0.1:8090/paddlespeech/vector
  [2022-08-01 09:01:27,093] [    INFO] - {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'vec': [1.4217487573623657, 5.626248836517334, -5.342073440551758, 1.177390217781067, 3.308061122894287, 1.7565997838974, 5.1678876876831055, 10.806346893310547, -3.822679042816162, -5.614130973815918, 2.6238481998443604, -0.8072965741157532, 1.963512659072876, -7.312864780426025, 0.011034967377781868, -9.723127365112305, 0.661963164806366, -6.976816654205322, 10.213465690612793, 7.494767189025879, 2.9105641841888428, 3.894925117492676, 3.7999846935272217, 7.106173992156982, 16.905324935913086, -7.149376392364502, 8.733112335205078, 3.423002004623413, -4.831653118133545, -11.403371810913086, 11.232216835021973, 7.127464771270752, -4.282831192016602, 2.4523589611053467, -5.13075065612793, -18.17765998840332, -2.611666440963745, -11.00034236907959, -6.731431007385254, 1.6564655303955078, 0.7618184685707092, 1.1253058910369873, -2.0838277339935303, 4.725739002227783, -8.782590866088867, -3.5398736000061035, 3.8142387866973877, 5.142062664031982, 2.162053346633911, 4.09642219543457, -6.416221618652344, 12.747454643249512, 1.9429889917373657, -15.152948379516602, 6.417416572570801, 16.097013473510742, -9.716649055480957, -1.9920448064804077, -3.364956855773926, -1.8719490766525269, 11.567351341247559, 3.6978795528411865, 11.258269309997559, 7.442364692687988, 9.183405876159668, 4.528151512145996, -1.2417811155319214, 4.395910263061523, 6.672768592834473, 5.889888763427734, 7.627115249633789, -0.6692016124725342, -11.889703750610352, -9.208883285522461, -7.427401542663574, -3.777655601501465, 6.917237758636475, -9.848749160766602, -2.094479560852051, -5.1351189613342285, 0.49564215540885925, 9.317541122436523, -5.9141845703125, -1.809845209121704, -0.11738205701112747, -7.169270992279053, -1.0578246116638184, -5.721685886383057, -5.117387294769287, 16.137670516967773, -4.473618984222412, 7.66243314743042, -0.5538089871406555, 9.631582260131836, -6.470466613769531, -8.54850959777832, 4.371622085571289, -0.7970349192619324, 4.479003429412842, -2.9758646488189697, 3.2721707820892334, 2.8382749557495117, 5.1345953941345215, -9.19078254699707, -0.5657423138618469, -4.874573230743408, 2.316561460494995, -5.984307289123535, -2.1798791885375977, 0.35541653633117676, -0.3178458511829376, 9.493547439575195, 2.114448070526123, 4.358088493347168, -12.089820861816406, 8.451695442199707, -7.925461769104004, 4.624246120452881, 4.428938388824463, 18.691999435424805, -2.620460033416748, -5.149182319641113, -0.3582168221473694, 8.488557815551758, 4.98148250579834, -9.326834678649902, -2.2544236183166504, 6.64176607131958, 1.2119656801223755, 10.977132797241211, 16.55504035949707, 3.323848247528076, 9.55185317993164, -1.6677050590515137, -0.7953923940658569, -8.605660438537598, -0.4735637903213501, 2.6741855144500732, -5.359188079833984, -2.6673784255981445, 0.6660736799240112, 15.443212509155273, 4.740597724914551, -3.4725306034088135, 11.592561721801758, -2.05450701713562, 1.7361239194869995, -8.26533031463623, -9.304476737976074, 5.406835079193115, -1.5180232524871826, -7.746610641479492, -6.089605331420898, 0.07112561166286469, -0.34904858469963074, -8.649889945983887, -9.998958587646484, -2.5648481845855713, -0.5399898886680603, 2.6018145084381104, -0.31927648186683655, -1.8815231323242188, -2.0721378326416016, -3.4105639457702637, -8.299802780151367, 1.4836379289627075, -15.366002082824707, -8.288193702697754, 3.884773015975952, -3.4876506328582764, 7.362995624542236, 0.4657321572303772, 3.1326000690460205, 12.438883781433105, -1.8337029218673706, 4.532927513122559, 2.726433277130127, 10.145345687866211, -6.521956920623779, 2.8971481323242188, -3.3925881385803223, 5.079156398773193, 7.759725093841553, 4.677562236785889, 5.8457818031311035, 2.4023921489715576, 7.707108974456787, 3.9711389541625977, -6.390035152435303, 6.126871109008789, -3.776031017303467, -11.118141174316406]}}
  [2022-08-01 09:01:27,094] [    INFO] - Response time 4.941739 s.
  ```

* Python API

  ``` python
  from paddlespeech.server.bin.paddlespeech_client import VectorClientExecutor
  import json

  vectorclient_executor = VectorClientExecutor()
  res = vectorclient_executor(
      input="85236145389.wav",
      server_ip="127.0.0.1",
      port=8090,
      task="spk")
  print(res.json())
  ```

  Output:

  ```text
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'vec': [1.4217487573623657, 5.626248836517334, -5.342073440551758, 1.177390217781067, 3.308061122894287, 1.7565997838974, 5.1678876876831055, 10.806346893310547, -3.822679042816162, -5.614130973815918, 2.6238481998443604, -0.8072965741157532, 1.963512659072876, -7.312864780426025, 0.011034967377781868, -9.723127365112305, 0.661963164806366, -6.976816654205322, 10.213465690612793, 7.494767189025879, 2.9105641841888428, 3.894925117492676, 3.7999846935272217, 7.106173992156982, 16.905324935913086, -7.149376392364502, 8.733112335205078, 3.423002004623413, -4.831653118133545, -11.403371810913086, 11.232216835021973, 7.127464771270752, -4.282831192016602, 2.4523589611053467, -5.13075065612793, -18.17765998840332, -2.611666440963745, -11.00034236907959, -6.731431007385254, 1.6564655303955078, 0.7618184685707092, 1.1253058910369873, -2.0838277339935303, 4.725739002227783, -8.782590866088867, -3.5398736000061035, 3.8142387866973877, 5.142062664031982, 2.162053346633911, 4.09642219543457, -6.416221618652344, 12.747454643249512, 1.9429889917373657, -15.152948379516602, 6.417416572570801, 16.097013473510742, -9.716649055480957, -1.9920448064804077, -3.364956855773926, -1.8719490766525269, 11.567351341247559, 3.6978795528411865, 11.258269309997559, 7.442364692687988, 9.183405876159668, 4.528151512145996, -1.2417811155319214, 4.395910263061523, 6.672768592834473, 5.889888763427734, 7.627115249633789, -0.6692016124725342, -11.889703750610352, -9.208883285522461, -7.427401542663574, -3.777655601501465, 6.917237758636475, -9.848749160766602, -2.094479560852051, -5.1351189613342285, 0.49564215540885925, 9.317541122436523, -5.9141845703125, -1.809845209121704, -0.11738205701112747, -7.169270992279053, -1.0578246116638184, -5.721685886383057, -5.117387294769287, 16.137670516967773, -4.473618984222412, 7.66243314743042, -0.5538089871406555, 9.631582260131836, -6.470466613769531, -8.54850959777832, 4.371622085571289, -0.7970349192619324, 4.479003429412842, -2.9758646488189697, 3.2721707820892334, 2.8382749557495117, 5.1345953941345215, -9.19078254699707, -0.5657423138618469, -4.874573230743408, 2.316561460494995, -5.984307289123535, -2.1798791885375977, 0.35541653633117676, -0.3178458511829376, 9.493547439575195, 2.114448070526123, 4.358088493347168, -12.089820861816406, 8.451695442199707, -7.925461769104004, 4.624246120452881, 4.428938388824463, 18.691999435424805, -2.620460033416748, -5.149182319641113, -0.3582168221473694, 8.488557815551758, 4.98148250579834, -9.326834678649902, -2.2544236183166504, 6.64176607131958, 1.2119656801223755, 10.977132797241211, 16.55504035949707, 3.323848247528076, 9.55185317993164, -1.6677050590515137, -0.7953923940658569, -8.605660438537598, -0.4735637903213501, 2.6741855144500732, -5.359188079833984, -2.6673784255981445, 0.6660736799240112, 15.443212509155273, 4.740597724914551, -3.4725306034088135, 11.592561721801758, -2.05450701713562, 1.7361239194869995, -8.26533031463623, -9.304476737976074, 5.406835079193115, -1.5180232524871826, -7.746610641479492, -6.089605331420898, 0.07112561166286469, -0.34904858469963074, -8.649889945983887, -9.998958587646484, -2.5648481845855713, -0.5399898886680603, 2.6018145084381104, -0.31927648186683655, -1.8815231323242188, -2.0721378326416016, -3.4105639457702637, -8.299802780151367, 1.4836379289627075, -15.366002082824707, -8.288193702697754, 3.884773015975952, -3.4876506328582764, 7.362995624542236, 0.4657321572303772, 3.1326000690460205, 12.438883781433105, -1.8337029218673706, 4.532927513122559, 2.726433277130127, 10.145345687866211, -6.521956920623779, 2.8971481323242188, -3.3925881385803223, 5.079156398773193, 7.759725093841553, 4.677562236785889, 5.8457818031311035, 2.4023921489715576, 7.707108974456787, 3.9711389541625977, -6.390035152435303, 6.126871109008789, -3.776031017303467, -11.118141174316406]}}
  ```

#### 7.2 Get the score between speaker audio embedding

**Note:** The response time will be slightly longer when using the client for the first time

- Command Line (Recommended)

  If `127.0.0.1` is not accessible, you need to use the actual service IP address.

  ``` bash
  paddlespeech_client vector --task score  --server_ip 127.0.0.1 --port 8090 --enroll 85236145389.wav --test 123456789.wav
  ```

  Usage:

  ``` bash
  paddlespeech_client vector --help
  ```

  Arguments:
    * server_ip: server ip. Default: 127.0.0.1
    * port: server port. Default: 8090
    * input(required): Input text to generate.
    * task: the task of vector, can be use 'spk' or 'score。If get the score, this must be 'score' parameter.
    * enroll: enroll audio
    * test: test audio
  
  Output:

  ```text
  [2022-08-01 09:04:42,275] [    INFO] - vector score http client start
  [2022-08-01 09:04:42,275] [    INFO] - enroll audio: 85236145389.wav, test audio: 123456789.wav
  [2022-08-01 09:04:42,275] [    INFO] - endpoint: http://127.0.0.1:8090/paddlespeech/vector/score
  [2022-08-01 09:04:44,611] [    INFO] - {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.4292638897895813}}
  [2022-08-01 09:04:44,611] [    INFO] - Response time 2.336258 s.
  ```

* Python API

  ``` python 
  from paddlespeech.server.bin.paddlespeech_client import VectorClientExecutor
  import json

  vectorclient_executor = VectorClientExecutor()
  res = vectorclient_executor(
      input=None,
      enroll_audio="85236145389.wav",
      test_audio="123456789.wav",
      server_ip="127.0.0.1",
      port=8090,
      task="score")
  print(res.json())
  ```

  Output:

  ```text
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.4292638897895813}}
  ```

### 8. Punctuation prediction
  
**Note:** The response time will be slightly longer when using the client for the first time

- Command Line (Recommended)

  If `127.0.0.1` is not accessible, you need to use the actual service IP address.

   ``` bash
   paddlespeech_client text --server_ip 127.0.0.1 --port 8090 --input "我认为跑步最重要的就是给我带来了身体健康"
   ```

  Usage:
  
  ```bash
  paddlespeech_client text --help
  ```
  Arguments:
  - `server_ip`: server ip. Default: 127.0.0.1
  - `port`: server port. Default: 8090
  - `input`(required): Input text to get punctuation.

  Output:
  ```text
  [2022-05-09 18:19:04,397] [    INFO] - The punc text: 我认为跑步最重要的就是给我带来了身体健康。
  [2022-05-09 18:19:04,397] [    INFO] - Response time 0.092407 s.
  ```

- Python API
  ```python
  from paddlespeech.server.bin.paddlespeech_client import TextClientExecutor

  textclient_executor = TextClientExecutor()
  res = textclient_executor(
      input="我认为跑步最重要的就是给我带来了身体健康",
      server_ip="127.0.0.1",
      port=8090,)
  print(res)
  ```

  Output:
  ```text
  我认为跑步最重要的就是给我带来了身体健康。
  ```

## Models supported by the service
### ASR model
Get all models supported by the ASR service via `paddlespeech_server stats --task asr`, where static models can be used for paddle inference inference.

### TTS model
Get all models supported by the TTS service via `paddlespeech_server stats --task tts`, where static models can be used for paddle inference inference.

### CLS model
Get all models supported by the CLS service via `paddlespeech_server stats --task cls`, where static models can be used for paddle inference inference.

### Vector model
Get all models supported by the TTS service via `paddlespeech_server stats --task vector`, where static models can be used for paddle inference inference.

### Text model
Get all models supported by the CLS service via `paddlespeech_server stats --task text`, where static models can be used for paddle inference inference.
