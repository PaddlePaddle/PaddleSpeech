#!/bin/bash

# http client test
# If `127.0.0.1` is not accessible, you need to use the actual service IP address.
paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol http --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.http.wav

# websocket client test
# If `127.0.0.1` is not accessible, you need to use the actual service IP address.
paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8192 --protocol websocket --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.ws.wav
