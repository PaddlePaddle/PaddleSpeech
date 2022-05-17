([简体中文](./README_cn.md)|English)

# Speech Server

## Introduction
This demo is an implementation of starting the voice service and accessing the service. It can be achieved with a single command using `paddlespeech_server` and `paddlespeech_client` or a few lines of code in python.


## Usage
### 1. Installation
see [installation](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

It is recommended to use **paddlepaddle 2.2.2** or above.
You can choose one way from meduim and hard to install paddlespeech.

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
  ```bash
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
  ```bash
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
  ```bash
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
  ```bash
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
    ```bash
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
  ```bash
  {'description': 'success.'}
  Save synthesized audio successfully on ./output.wav.
  Audio duration: 3.612500 s.

  ```

### 6. CLS Client Usage
**Note:** The response time will be slightly longer when using the client for the first time
- Command Line (Recommended)

   If `127.0.0.1` is not accessible, you need to use the actual service IP address.

   ```
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
  ```bash
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
  ```bash
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

  * Usage:

  ``` bash
  paddlespeech_client vector --help
  ```

  * Arguments:
    * server_ip: server ip. Default: 127.0.0.1
    * port: server port. Default: 8090
    * input(required): Input text to generate.
    * task: the task of vector, can be use 'spk' or 'score。Default is 'spk'。
    * enroll: enroll audio
    * test: test audio

  * Output:

  ``` bash
    [2022-05-08 00:18:44,249] [    INFO] - vector http client start
    [2022-05-08 00:18:44,250] [    INFO] - the input audio: 85236145389.wav
    [2022-05-08 00:18:44,250] [    INFO] - endpoint: http://127.0.0.1:8090/paddlespeech/vector
    [2022-05-08 00:18:44,250] [    INFO] - http://127.0.0.1:8590/paddlespeech/vector
    [2022-05-08 00:18:44,406] [    INFO] - The vector: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'vec': [1.421751856803894, 5.626245498657227, -5.342077255249023, 1.1773887872695923, 3.3080549240112305, 1.7565933465957642, 5.167886257171631, 10.806358337402344, -3.8226819038391113, -5.614140033721924, 2.6238479614257812, -0.8072972893714905, 1.9635076522827148, -7.312870025634766, 0.011035939678549767, -9.723129272460938, 0.6619706153869629, -6.976806163787842, 10.213476181030273, 7.494769096374512, 2.9105682373046875, 3.8949244022369385, 3.799983501434326, 7.106168746948242, 16.90532875061035, -7.149388313293457, 8.733108520507812, 3.423006296157837, -4.831653594970703, -11.403363227844238, 11.232224464416504, 7.127461910247803, -4.282842636108398, 2.452359437942505, -5.130749702453613, -18.17766761779785, -2.6116831302642822, -11.000344276428223, -6.731433391571045, 1.6564682722091675, 0.7618281245231628, 1.125300407409668, -2.0838370323181152, 4.725743293762207, -8.782588005065918, -3.5398752689361572, 3.8142364025115967, 5.142068862915039, 2.1620609760284424, 4.09643030166626, -6.416214942932129, 12.747446060180664, 1.9429892301559448, -15.15294361114502, 6.417416095733643, 16.09701156616211, -9.716667175292969, -1.9920575618743896, -3.36494779586792, -1.8719440698623657, 11.567351341247559, 3.6978814601898193, 11.258262634277344, 7.442368507385254, 9.183408737182617, 4.528149127960205, -1.2417854070663452, 4.395912170410156, 6.6727728843688965, 5.88988733291626, 7.627128601074219, -0.6691966652870178, -11.889698028564453, -9.20886516571045, -7.42740535736084, -3.777663230895996, 6.917238712310791, -9.848755836486816, -2.0944676399230957, -5.1351165771484375, 0.4956451654434204, 9.317537307739258, -5.914181232452393, -1.809860348701477, -0.11738915741443634, -7.1692705154418945, -1.057827353477478, -5.721670627593994, -5.117385387420654, 16.13765525817871, -4.473617076873779, 7.6624321937561035, -0.55381840467453, 9.631585121154785, -6.470459461212158, -8.548508644104004, 4.371616840362549, -0.7970245480537415, 4.4789886474609375, -2.975860834121704, 3.2721822261810303, 2.838287830352783, 5.134591102600098, -9.19079875946045, -0.5657302737236023, -4.8745832443237305, 2.3165574073791504, -5.984319686889648, -2.1798853874206543, 0.3554139733314514, -0.3178512752056122, 9.493552207946777, 2.1144471168518066, 4.358094692230225, -12.089824676513672, 8.451693534851074, -7.925466537475586, 4.624246597290039, 4.428936958312988, 18.69200897216797, -2.6204581260681152, -5.14918851852417, -0.3582090139389038, 8.488558769226074, 4.98148775100708, -9.326835632324219, -2.2544219493865967, 6.641760349273682, 1.2119598388671875, 10.977124214172363, 16.555034637451172, 3.3238420486450195, 9.551861763000488, -1.6676981449127197, -0.7953944206237793, -8.605667114257812, -0.4735655188560486, 2.674196243286133, -5.359177112579346, -2.66738224029541, 0.6660683155059814, 15.44322681427002, 4.740593433380127, -3.472534418106079, 11.592567443847656, -2.0544962882995605, 1.736127495765686, -8.265326499938965, -9.30447769165039, 5.406829833984375, -1.518022894859314, -7.746612548828125, -6.089611053466797, 0.07112743705511093, -0.3490503430366516, -8.64989185333252, -9.998957633972168, -2.564845085144043, -0.5399947762489319, 2.6018123626708984, -0.3192799389362335, -1.8815255165100098, -2.0721492767333984, -3.410574436187744, -8.29980754852295, 1.483638048171997, -15.365986824035645, -8.288211822509766, 3.884779930114746, -3.4876468181610107, 7.362999439239502, 0.4657334089279175, 3.1326050758361816, 12.438895225524902, -1.8337041139602661, 4.532927989959717, 2.7264339923858643, 10.14534854888916, -6.521963596343994, 2.897155523300171, -3.392582654953003, 5.079153060913086, 7.7597246170043945, 4.677570819854736, 5.845779895782471, 2.402411460876465, 7.7071051597595215, 3.9711380004882812, -6.39003849029541, 6.12687873840332, -3.776029348373413, -11.118121147155762]}}
    [2022-05-08 00:18:44,406] [    INFO] - Response time 0.156481 s.
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

* Output:

  ``` bash
    {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'vec': [1.421751856803894, 5.626245498657227, -5.342077255249023, 1.1773887872695923, 3.3080549240112305, 1.7565933465957642, 5.167886257171631, 10.806358337402344, -3.8226819038391113, -5.614140033721924, 2.6238479614257812, -0.8072972893714905, 1.9635076522827148, -7.312870025634766, 0.011035939678549767, -9.723129272460938, 0.6619706153869629, -6.976806163787842, 10.213476181030273, 7.494769096374512, 2.9105682373046875, 3.8949244022369385, 3.799983501434326, 7.106168746948242, 16.90532875061035, -7.149388313293457, 8.733108520507812, 3.423006296157837, -4.831653594970703, -11.403363227844238, 11.232224464416504, 7.127461910247803, -4.282842636108398, 2.452359437942505, -5.130749702453613, -18.17766761779785, -2.6116831302642822, -11.000344276428223, -6.731433391571045, 1.6564682722091675, 0.7618281245231628, 1.125300407409668, -2.0838370323181152, 4.725743293762207, -8.782588005065918, -3.5398752689361572, 3.8142364025115967, 5.142068862915039, 2.1620609760284424, 4.09643030166626, -6.416214942932129, 12.747446060180664, 1.9429892301559448, -15.15294361114502, 6.417416095733643, 16.09701156616211, -9.716667175292969, -1.9920575618743896, -3.36494779586792, -1.8719440698623657, 11.567351341247559, 3.6978814601898193, 11.258262634277344, 7.442368507385254, 9.183408737182617, 4.528149127960205, -1.2417854070663452, 4.395912170410156, 6.6727728843688965, 5.88988733291626, 7.627128601074219, -0.6691966652870178, -11.889698028564453, -9.20886516571045, -7.42740535736084, -3.777663230895996, 6.917238712310791, -9.848755836486816, -2.0944676399230957, -5.1351165771484375, 0.4956451654434204, 9.317537307739258, -5.914181232452393, -1.809860348701477, -0.11738915741443634, -7.1692705154418945, -1.057827353477478, -5.721670627593994, -5.117385387420654, 16.13765525817871, -4.473617076873779, 7.6624321937561035, -0.55381840467453, 9.631585121154785, -6.470459461212158, -8.548508644104004, 4.371616840362549, -0.7970245480537415, 4.4789886474609375, -2.975860834121704, 3.2721822261810303, 2.838287830352783, 5.134591102600098, -9.19079875946045, -0.5657302737236023, -4.8745832443237305, 2.3165574073791504, -5.984319686889648, -2.1798853874206543, 0.3554139733314514, -0.3178512752056122, 9.493552207946777, 2.1144471168518066, 4.358094692230225, -12.089824676513672, 8.451693534851074, -7.925466537475586, 4.624246597290039, 4.428936958312988, 18.69200897216797, -2.6204581260681152, -5.14918851852417, -0.3582090139389038, 8.488558769226074, 4.98148775100708, -9.326835632324219, -2.2544219493865967, 6.641760349273682, 1.2119598388671875, 10.977124214172363, 16.555034637451172, 3.3238420486450195, 9.551861763000488, -1.6676981449127197, -0.7953944206237793, -8.605667114257812, -0.4735655188560486, 2.674196243286133, -5.359177112579346, -2.66738224029541, 0.6660683155059814, 15.44322681427002, 4.740593433380127, -3.472534418106079, 11.592567443847656, -2.0544962882995605, 1.736127495765686, -8.265326499938965, -9.30447769165039, 5.406829833984375, -1.518022894859314, -7.746612548828125, -6.089611053466797, 0.07112743705511093, -0.3490503430366516, -8.64989185333252, -9.998957633972168, -2.564845085144043, -0.5399947762489319, 2.6018123626708984, -0.3192799389362335, -1.8815255165100098, -2.0721492767333984, -3.410574436187744, -8.29980754852295, 1.483638048171997, -15.365986824035645, -8.288211822509766, 3.884779930114746, -3.4876468181610107, 7.362999439239502, 0.4657334089279175, 3.1326050758361816, 12.438895225524902, -1.8337041139602661, 4.532927989959717, 2.7264339923858643, 10.14534854888916, -6.521963596343994, 2.897155523300171, -3.392582654953003, 5.079153060913086, 7.7597246170043945, 4.677570819854736, 5.845779895782471, 2.402411460876465, 7.7071051597595215, 3.9711380004882812, -6.39003849029541, 6.12687873840332, -3.776029348373413, -11.118121147155762]}}
  ```

#### 7.2 Get the score between speaker audio embedding

**Note:** The response time will be slightly longer when using the client for the first time

- Command Line (Recommended)

  If `127.0.0.1` is not accessible, you need to use the actual service IP address.

  ``` bash
  paddlespeech_client vector --task score  --server_ip 127.0.0.1 --port 8090 --enroll 85236145389.wav --test 123456789.wav
  ```

  * Usage:

  ``` bash
  paddlespeech_client vector --help
  ```

  * Arguments:
    * server_ip: server ip. Default: 127.0.0.1
    * port: server port. Default: 8090
    * input(required): Input text to generate.
    * task: the task of vector, can be use 'spk' or 'score。If get the score, this must be 'score' parameter.
    * enroll: enroll audio
    * test: test audio
  
* Output:

``` bash
  [2022-05-09 10:28:40,556] [    INFO] - vector score http client start
  [2022-05-09 10:28:40,556] [    INFO] - enroll audio: 85236145389.wav, test audio: 123456789.wav
  [2022-05-09 10:28:40,556] [    INFO] - endpoint: http://127.0.0.1:8090/paddlespeech/vector/score
  [2022-05-09 10:28:40,731] [    INFO] - The vector score is: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.4292638897895813}}
  [2022-05-09 10:28:40,731] [    INFO] - The vector: None
  [2022-05-09 10:28:40,731] [    INFO] - Response time 0.175514 s.
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

* Output:

``` bash
[2022-05-09 10:34:54,769] [    INFO] - vector score http client start
[2022-05-09 10:34:54,771] [    INFO] - enroll audio: 85236145389.wav, test audio: 123456789.wav
[2022-05-09 10:34:54,771] [    INFO] - endpoint: http://127.0.0.1:8090/paddlespeech/vector/score
[2022-05-09 10:34:55,026] [    INFO] - The vector score is: {'success': True, 'code': 200, 'message': {'description': 'success'}, 'result': {'score': 0.4292638897895813}}
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
  参数:
  - `server_ip`: server ip. Default: 127.0.0.1
  - `port`: server port. Default: 8090
  - `input`(required): Input text to get punctuation.

  Output:
  ```bash
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
  ```bash
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
