# PaddleSpeech Server 命令行工具

(简体中文|[English](./README.md))

它提供了最简便的方式调用 PaddleSpeech 语音服务用一行命令就可以轻松启动服务和调用服务。

 ## 服务端命令行使用
 ### 帮助
 ```bash
 paddlespeech_server help
 ```
 ### 启动服务
 首先设置服务相关配置文件，类似于 `./paddlespeech/server/conf/application.yaml`，同时设置服务配置中的语音任务模型相关配置，类似于 `./paddlespeech/server/conf/tts/tts.yaml`。
 然后启动服务：
 ```bash
 paddlespeech_server start --config_file ./paddlespeech/server/conf/application.yaml
 ```

 ## 客户端命令行使用
 ### 帮助
 ```bash
 paddlespeech_client help
 ```
 ### 访问语音识别服务 
 ```
 paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input /paddlespeech/server/tests/16_audio.wav
 ```
 
 ### 访问语音合成服务
 ```bash
 paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```
