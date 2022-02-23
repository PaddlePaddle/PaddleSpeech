# PaddleSpeech Server Command Line

([简体中文](./README_cn.md)|English)

 The simplest approach to use PaddleSpeech Server including server and client.

 ## PaddleSpeech Server
 ### Help
 ```bash
 paddlespeech_server help
 ```
 ### Start the server
 First set the service-related configuration parameters, similar to `./paddlespeech/server/conf/application.yaml`,
 Then start the service:
 ```bash
 paddlespeech_server start --config_file ./paddlespeech/server/conf/application.yaml
 ```

 ## PaddleSpeech Client
 ### Help
 ```bash
 paddlespeech_client help
 ```
 ### Access speech recognition services 
 ```
 paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input ./paddlespeech/server/tests/16_audio.wav
 ```
 
 ### Access text to speech services
 ```bash
 paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```
 
