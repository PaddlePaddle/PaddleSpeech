(简体中文|[English](./README.md))

# 流式语音合成服务

## 介绍
本文介绍了使用FastDeploy搭建流式语音合成服务的方法。

`服务端`必须在docker内启动,而`客户端`不是必须在docker容器内.

我们假设你的模型和代码(`服务端`会加载模型和代码以启动服务)在你主机上的绝对路径是`$PWD`,模型和代码在docker内的绝对路径是`/models`

## 使用
### 1. 服务端
#### 1.1 Docker

`docker pull registry.baidubce.com/paddlepaddle/fastdeploy_serving_cpu_only:22.09`

`docker run -dit  --net=host --name fastdeploy --shm-size="1g" -v $PWD:/models registry.baidubce.com/paddlepaddle/fastdeploy_serving_cpu_only:22.09`

`docker exec -it -u root fastdeploy bash`

#### 1.2 安装(在docker内)

`apt-get install build-essential python3-dev libssl-dev libffi-dev libxml2 libxml2-dev libxslt1-dev zlib1g-dev libsndfile1 language-pack-zh-hans wget zip`

`pip3 install paddlespeech`

`export LC_ALL="zh_CN.UTF-8"`

`export LANG="zh_CN.UTF-8"`

`export LANGUAGE="zh_CN:zh:en_US:en"`

#### 1.3 下载模型(在docker内)

`cd /models/streaming_tts_serving/1`

`wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip`

`wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_onnx_0.2.0.zip`

`unzip fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip`

`unzip mb_melgan_csmsc_onnx_0.2.0.zip`

#### 1.4 启动服务端(在docker内)

`fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=streaming_tts_serving`

**服务启动的默认端口是8000(for http),8001(for grpc),8002(for metrics). 如果想要改变服务的端口号,在上述命令后面添加以下参数即可`--http-port 9000 --grpc-port 9001 --metrics-port 9002`**

### 2. 客户端
#### 2.1 安装

`pip3 install tritonclient[all]`

#### 2.2 发送请求

`python3 /models/streaming_tts_serving/stream_client.py`

