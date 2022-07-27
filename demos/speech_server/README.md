([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the voice service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.

For service interface definition, please check:
- [PaddleSpeech Server RESTful API](https://github.com/PaddlePaddle/PaddleSpeech/wiki/PaddleSpeech-Server-RESTful-API)


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.3.1** or above.

You can choose one way from easy, meduim and hard to install paddlespeech.

**If you install in easy mode, you need to prepare the yaml file by yourself, you can refer to the yaml file in the conf directory.**

### 2. Prepare config File
The configuration file can be found in `conf/application.yaml` .
Among them, `engine_list` indicates the speech engine that will be included in the service to be started, in the format of `<speech task>_<engine type>`.
At present, the speech tasks integrated by the service include: asr (speech recognition), tts (text to sppech) and cls (audio classification).
Currently the engine type supports two forms: python and inference (Paddle Inference)
**Note:** If the service can be started normally in the container, but the client access IP is unreachable, you can try to replace the `host` address in the configuration file with the local IP address.


The input of  ASR client demo should be a WAV file(`.wav`), and the sample rate must be the same as the model.

Here are sample files for thisASR client demo that can be downloaded:
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav
```

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
  [2022-02-23 18:11:22,819] [    INFO] - {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '我认为跑步最重要的就是给我带来了身体健康'}}
  [2022-02-23 18:11:22,820] [    INFO] - time cost 0.689145 s.

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
  ```text
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'transcription': '我认为跑步最重要的就是给我带来了身体健康'}}
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
    [2022-02-23 15:20:37,875] [    INFO] - {'description': 'success.'}
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
  [2022-05-25 12:25:36,165] [    INFO] - vector http client start
  [2022-05-25 12:25:36,165] [    INFO] - the input audio: 85236145389.wav
  [2022-05-25 12:25:36,165] [    INFO] - endpoint: http://127.0.0.1:8790/paddlespeech/vector
  [2022-05-25 12:25:36,166] [    INFO] - http://127.0.0.1:8790/paddlespeech/vector
  [2022-05-25 12:25:36,324] [    INFO] - The vector: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'vec': [-1.3251205682754517, 7.860682487487793, -4.620625972747803, 0.3000721037387848, 2.2648534774780273, -1.1931440830230713, 3.064713716506958, 7.673594951629639, -6.004472732543945, -12.024259567260742, -1.9496068954467773, 3.126953601837158, 1.6188379526138306, -7.638310432434082, -1.2299772500991821, -12.33833122253418, 2.1373026371002197, -5.395712375640869, 9.717328071594238, 5.675230503082275, 3.7805123329162598, 3.0597171783447266, 3.429692029953003, 8.9760103225708, 13.174124717712402, -0.5313228368759155, 8.942471504211426, 4.465109825134277, -4.426247596740723, -9.726503372192383, 8.399328231811523, 7.223917484283447, -7.435853958129883, 2.9441683292388916, -4.343039512634277, -13.886964797973633, -1.6346734762191772, -10.902740478515625, -5.311244964599609, 3.800722122192383, 3.897603750228882, -2.123077392578125, -2.3521194458007812, 4.151031017303467, -7.404866695404053, 0.13911646604537964, 2.4626107215881348, 4.96645450592041, 0.9897574186325073, 5.483975410461426, -3.3574001789093018, 10.13400650024414, -0.6120170950889587, -10.403095245361328, 4.600754261016846, 16.009349822998047, -7.78369140625, -4.194530487060547, -6.93686056137085, 1.1789555549621582, 11.490800857543945, 4.23802375793457, 9.550930976867676, 8.375045776367188, 7.508914470672607, -0.6570729613304138, -0.3005157709121704, 2.8406054973602295, 3.0828027725219727, 0.7308170199394226, 6.1483540534973145, 0.1376611888408661, -13.424735069274902, -7.746140480041504, -2.322798252105713, -8.305252075195312, 2.98791241645813, -10.99522876739502, 0.15211068093776703, -2.3820347785949707, -1.7984174489974976, 8.49562931060791, -5.852236747741699, -3.755497932434082, 0.6989710927009583, -5.270299434661865, -2.6188621520996094, -1.8828465938568115, -4.6466498374938965, 14.078543663024902, -0.5495333075523376, 10.579157829284668, -3.216050148010254, 9.349003791809082, -4.381077766418457, -11.675816535949707, -2.863020658493042, 4.5721755027771, 2.246612071990967, -4.574341773986816, 1.8610187768936157, 2.3767874240875244, 5.625787734985352, -9.784077644348145, 0.6496725678443909, -1.457950472831726, 0.4263263940811157, -4.921126365661621, -2.4547839164733887, 3.4869801998138428, -0.4265422224998474, 8.341268539428711, 1.356552004814148, 7.096688270568848, -13.102828979492188, 8.01673412322998, -7.115934371948242, 1.8699780702590942, 0.20872099697589874, 14.699383735656738, -1.0252779722213745, -2.6107232570648193, -2.5082311630249023, 8.427192687988281, 6.913852691650391, -6.29124641418457, 0.6157366037368774, 2.489687919616699, -3.4668266773223877, 9.92176342010498, 11.200815200805664, -0.19664029777050018, 7.491600513458252, -0.6231271624565125, -0.2584814429283142, -9.947997093200684, -0.9611040949821472, 1.1649218797683716, -2.1907122135162354, -1.502848744392395, -0.5192610621452332, 15.165953636169434, 2.4649462699890137, -0.998044490814209, 7.44166374206543, -2.0768048763275146, 3.5896823406219482, -7.305543422698975, -7.562084674835205, 4.32333517074585, 0.08044180274009705, -6.564010143280029, -2.314805269241333, -1.7642345428466797, -2.470881700515747, -7.6756181716918945, -9.548877716064453, -1.017755389213562, 0.1698644608259201, 2.5877134799957275, -1.8752295970916748, -0.36614322662353516, -6.049378395080566, -2.3965611457824707, -5.945338726043701, 0.9424033164978027, -13.155974388122559, -7.45780086517334, 0.14658108353614807, -3.7427968978881836, 5.841492652893066, -1.2872905731201172, 5.569431304931641, 12.570590019226074, 1.0939218997955322, 2.2142086029052734, 1.9181575775146484, 6.991420745849609, -5.888138771057129, 3.1409823894500732, -2.0036280155181885, 2.4434285163879395, 9.973138809204102, 5.036680221557617, 2.005120277404785, 2.861560344696045, 5.860223770141602, 2.917618751525879, -1.63111412525177, 2.0292205810546875, -4.070415019989014, -6.831437110900879]}}
  [2022-05-25 12:25:36,324] [    INFO] - Response time 0.159053 s.
  ```

* Python API

  ``` python
  from paddlespeech.server.bin.paddlespeech_client import VectorClientExecutor

  vectorclient_executor = VectorClientExecutor()
  res = vectorclient_executor(
      input="85236145389.wav",
      server_ip="127.0.0.1",
      port=8090,
      task="spk")
  print(res)
  ```

  Output:

  ```text
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'vec': [-1.3251205682754517, 7.860682487487793, -4.620625972747803, 0.3000721037387848, 2.2648534774780273, -1.1931440830230713, 3.064713716506958, 7.673594951629639, -6.004472732543945, -12.024259567260742, -1.9496068954467773, 3.126953601837158, 1.6188379526138306, -7.638310432434082, -1.2299772500991821, -12.33833122253418, 2.1373026371002197, -5.395712375640869, 9.717328071594238, 5.675230503082275, 3.7805123329162598, 3.0597171783447266, 3.429692029953003, 8.9760103225708, 13.174124717712402, -0.5313228368759155, 8.942471504211426, 4.465109825134277, -4.426247596740723, -9.726503372192383, 8.399328231811523, 7.223917484283447, -7.435853958129883, 2.9441683292388916, -4.343039512634277, -13.886964797973633, -1.6346734762191772, -10.902740478515625, -5.311244964599609, 3.800722122192383, 3.897603750228882, -2.123077392578125, -2.3521194458007812, 4.151031017303467, -7.404866695404053, 0.13911646604537964, 2.4626107215881348, 4.96645450592041, 0.9897574186325073, 5.483975410461426, -3.3574001789093018, 10.13400650024414, -0.6120170950889587, -10.403095245361328, 4.600754261016846, 16.009349822998047, -7.78369140625, -4.194530487060547, -6.93686056137085, 1.1789555549621582, 11.490800857543945, 4.23802375793457, 9.550930976867676, 8.375045776367188, 7.508914470672607, -0.6570729613304138, -0.3005157709121704, 2.8406054973602295, 3.0828027725219727, 0.7308170199394226, 6.1483540534973145, 0.1376611888408661, -13.424735069274902, -7.746140480041504, -2.322798252105713, -8.305252075195312, 2.98791241645813, -10.99522876739502, 0.15211068093776703, -2.3820347785949707, -1.7984174489974976, 8.49562931060791, -5.852236747741699, -3.755497932434082, 0.6989710927009583, -5.270299434661865, -2.6188621520996094, -1.8828465938568115, -4.6466498374938965, 14.078543663024902, -0.5495333075523376, 10.579157829284668, -3.216050148010254, 9.349003791809082, -4.381077766418457, -11.675816535949707, -2.863020658493042, 4.5721755027771, 2.246612071990967, -4.574341773986816, 1.8610187768936157, 2.3767874240875244, 5.625787734985352, -9.784077644348145, 0.6496725678443909, -1.457950472831726, 0.4263263940811157, -4.921126365661621, -2.4547839164733887, 3.4869801998138428, -0.4265422224998474, 8.341268539428711, 1.356552004814148, 7.096688270568848, -13.102828979492188, 8.01673412322998, -7.115934371948242, 1.8699780702590942, 0.20872099697589874, 14.699383735656738, -1.0252779722213745, -2.6107232570648193, -2.5082311630249023, 8.427192687988281, 6.913852691650391, -6.29124641418457, 0.6157366037368774, 2.489687919616699, -3.4668266773223877, 9.92176342010498, 11.200815200805664, -0.19664029777050018, 7.491600513458252, -0.6231271624565125, -0.2584814429283142, -9.947997093200684, -0.9611040949821472, 1.1649218797683716, -2.1907122135162354, -1.502848744392395, -0.5192610621452332, 15.165953636169434, 2.4649462699890137, -0.998044490814209, 7.44166374206543, -2.0768048763275146, 3.5896823406219482, -7.305543422698975, -7.562084674835205, 4.32333517074585, 0.08044180274009705, -6.564010143280029, -2.314805269241333, -1.7642345428466797, -2.470881700515747, -7.6756181716918945, -9.548877716064453, -1.017755389213562, 0.1698644608259201, 2.5877134799957275, -1.8752295970916748, -0.36614322662353516, -6.049378395080566, -2.3965611457824707, -5.945338726043701, 0.9424033164978027, -13.155974388122559, -7.45780086517334, 0.14658108353614807, -3.7427968978881836, 5.841492652893066, -1.2872905731201172, 5.569431304931641, 12.570590019226074, 1.0939218997955322, 2.2142086029052734, 1.9181575775146484, 6.991420745849609, -5.888138771057129, 3.1409823894500732, -2.0036280155181885, 2.4434285163879395, 9.973138809204102, 5.036680221557617, 2.005120277404785, 2.861560344696045, 5.860223770141602, 2.917618751525879, -1.63111412525177, 2.0292205810546875, -4.070415019989014, -6.831437110900879]}}
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
  [2022-05-25 12:33:24,527] [    INFO] - vector score http client start
  [2022-05-25 12:33:24,527] [    INFO] - enroll audio: 85236145389.wav, test audio: 123456789.wav
  [2022-05-25 12:33:24,528] [    INFO] - endpoint: http://127.0.0.1:8790/paddlespeech/vector/score
  [2022-05-25 12:33:24,695] [    INFO] - The vector score is: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.45332613587379456}}
  [2022-05-25 12:33:24,696] [    INFO] - The vector: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.45332613587379456}}
  [2022-05-25 12:33:24,696] [    INFO] - Response time 0.168271 s.
  ```

* Python API

  ``` python 
  from paddlespeech.server.bin.paddlespeech_client import VectorClientExecutor

  vectorclient_executor = VectorClientExecutor()
  res = vectorclient_executor(
      input=None,
      enroll_audio="85236145389.wav",
      test_audio="123456789.wav",
      server_ip="127.0.0.1",
      port=8090,
      task="score")
  print(res)
  ```

  Output:

  ```text
  [2022-05-25 12:30:14,143] [    INFO] - vector score http client start
  [2022-05-25 12:30:14,143] [    INFO] - enroll audio: 85236145389.wav, test audio: 123456789.wav
  [2022-05-25 12:30:14,143] [    INFO] - endpoint: http://127.0.0.1:8790/paddlespeech/vector/score
  [2022-05-25 12:30:14,363] [    INFO] - The vector score is: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.45332613587379456}}
  {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.45332613587379456}}
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
