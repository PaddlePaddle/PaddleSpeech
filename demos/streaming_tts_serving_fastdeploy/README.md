([简体中文](./README_cn.md)|English)

# Streaming Speech Synthesis Service

## Introduction
This demo is an implementation of starting the streaming speech synthesis service and accessing the service.

`Server` must be started in the docker, while `Client` does not have to be in the docker.

We assume your model and code (which will be loaded by the `Server`) absolute path in your host is `$PWD` and the model absolute path in docker is `/models`

## Usage
### 1. Server
#### 1.1 Docker

`docker pull registry.baidubce.com/paddlepaddle/fastdeploy_serving_cpu_only:22.09`

`docker run -dit  --net=host --name fastdeploy --shm-size="1g" -v $PWD:/models registry.baidubce.com/paddlepaddle/fastdeploy_serving_cpu_only:22.09`

`docker exec -it -u root fastdeploy bash`

#### 1.2 Installation(inside the docker)

`apt-get install build-essential python3-dev libssl-dev libffi-dev libxml2 libxml2-dev libxslt1-dev zlib1g-dev libsndfile1 language-pack-zh-hans wget zip`

`pip3 install paddlespeech`

`export LC_ALL="zh_CN.UTF-8"`

`export LANG="zh_CN.UTF-8"`

`export LANGUAGE="zh_CN:zh:en_US:en"`

#### 1.3 Download models(inside the docker)

`cd /models/streaming_tts_serving/1`

`wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip`

`wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_onnx_0.2.0.zip`

`unzip fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip`

`unzip mb_melgan_csmsc_onnx_0.2.0.zip`

#### 1.4 Start the server(inside the docker)

`fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=streaming_tts_serving`

**The default port is 8000(for http),8001(for grpc),8002(for metrics). If you want to change the port, add the command `--http-port 9000 --grpc-port 9001 --metrics-port 9002`**

### 2. Client
#### 2.1 Installation

`pip3 install tritonclient[all]`

#### 2.2 Send request

`python3 /models/streaming_tts_serving/stream_client.py`

