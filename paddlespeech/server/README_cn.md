# PaddleSpeech Server 命令行工具

(简体中文|[English](./README.md))

它提供了最简便的方式调用 PaddleSpeech 语音服务用一行命令就可以轻松启动服务和调用服务。

 ## 服务端命令行使用
 ### 帮助
 ```bash
 paddlespeech_server help
 ```
 ### 启动服务
 首先设置服务相关配置文件，类似于 `./conf/application.yaml`，设置 `engine_list`，该值表示即将启动的服务中包含的语音任务。
 **注意：** 如果在容器里可正常启动服务，但客户端访问 ip 不可达，可尝试将配置文件中 `host` 地址换成本地 ip 地址。
 然后启动服务：
 ```bash
 paddlespeech_server start --config_file ./conf/application.yaml
 ```

 ## 客户端命令行使用
 ### 帮助
 ```bash
 paddlespeech_client help
 ```
 ### 访问语音识别服务 
 ```
 paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
 ```
 
 ### 访问语音合成服务
 ```bash
 paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```

 ### 访问音频分类服务
 ```bash
 paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input input.wav
 ```

## 流式ASR

### 启动流式语音识别服务

```
paddlespeech_server start --config_file conf/ws_conformer_application.yaml
```

### 访问流式语音识别服务

```
paddlespeech_client asr_online  --server_ip 127.0.0.1 --port 8090 --input zh.wav
```

## 流式TTS

### 启动流式语音合成服务

```
paddlespeech_server start --config_file conf/tts_online_application.yaml
```

### 访问流式语音合成服务

```
paddlespeech_client tts_online  --server_ip 127.0.0.1 --port 8092 --input "您好，欢迎使用百度飞桨深度学习框架！" --output output.wav
```

## 声纹识别

### 启动声纹识别服务

```
paddlespeech_server start --config_file conf/vector_application.yaml
```

### 获取说话人音频声纹

```
paddlespeech_client vector --task spk  --server_ip 127.0.0.1 --port 8090 --input 85236145389.wav
```

### 两个说话人音频声纹打分

```
paddlespeech_client vector --task score  --server_ip 127.0.0.1 --port 8090 --enroll 123456789.wav --test 85236145389.wav
```