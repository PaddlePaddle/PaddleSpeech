# PaddleSpeech Server Command Line

([简体中文](./README_cn.md)|English)

 The simplest approach to use PaddleSpeech Server including server and client.

 ## PaddleSpeech Server
 ### Help
 ```bash
 paddlespeech_server help
 ```
 ### Start the server
 First set the service-related configuration parameters, similar to `./conf/application.yaml`. Set `engine_list`, which represents the speech tasks included in the service to be started
 Then start the service:
 ```bash
 paddlespeech_server start --config_file ./conf/application.yaml
 ```

 ## PaddleSpeech Client
 ### Help
 ```bash
 paddlespeech_client help
 ```
 ### Access speech recognition services 
 ```
 paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
 ```
 
 ### Access text to speech services
 ```bash
 paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```
 
 ### Access audio classification services
 ```bash
 paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input input.wav
 ```

 ## Online ASR Server

### Lanuch online asr server
```
paddlespeech_server start --config_file conf/ws_conformer_application.yaml
```

### Access online asr server

```
paddlespeech_client asr_online  --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
```

## Online TTS Server

### Lanuch online tts server
```
paddlespeech_server start --config_file conf/tts_online_application.yaml
```

### Access online tts server

```
paddlespeech_client tts_online  --server_ip 127.0.0.1 --port 8092 --input "您好，欢迎使用百度飞桨深度学习框架！" --output output.wav
```
