(简体中文|[English](./README.md))

# 定制化语音识别演示
## 介绍
定制化的语音识别是满足一些特定场景的语句识别的技术。

可以参见简单的原理教程：
https://aistudio.baidu.com/aistudio/projectdetail/3986429

这个 demo 是打车报销单的场景识别，定制化了地点。

## 使用方法
### 1. 配置环境
安装paddle:2.2.2 docker镜像。
```
sudo nvidia-docker run --privileged  --net=host --ipc=host -it --rm -v $PWD:/paddle --name=paddle_demo_docker registry.baidubce.com/paddlepaddle/paddle:2.2.2 /bin/bash 
```

### 2. 演示
* 运行如下命令，完成相关资源和库的下载和服务启动。
```
bash websocket_server.sh
```
上面脚本完成了如下两个功能：
1. 完成resource.tar.gz下载，解压后,会在resource中发现如下目录：
model: 声学模型
graph: 解码构图
lib: 相关库
bin: 运行程序
data: 语音数据

2. 通过websocket_server_main来启动服务。
这里简单的介绍几个参数:
port是服务端口，
graph_path用来指定解码图文件，
model相关参数用来指定声学模型文件。
其他参数说明可参见代码：
PaddleSpeech/speechx/speechx/decoder/param.h
PaddleSpeech/speechx/examples/ds2_ol/websocket/websocket_server_main.cc

* 在另一个终端中， 通过client发送数据，得到结果。运行如下命令：
```
bash websocket_client.sh
```
通过websocket_client_main来启动client服务，其中$wav_scp是发送的语音句子集合，port为服务端口。

* 结果：
client的log中可以看到如下类似的结果
```
0513 10:58:13.827821 41768 recognizer_test_main.cc:56] wav len (sample): 70208
I0513 10:58:13.884493 41768 feature_cache.h:52] set finished
I0513 10:58:24.247171 41768 paddle_nnet.h:76] Tensor neml: 10240
I0513 10:58:24.247249 41768 paddle_nnet.h:76] Tensor neml: 10240
LOG ([5.5.544~2-f21d7]:main():decoder/recognizer_test_main.cc:90)  the result of case_10 is 五月十二日二十二点三十六分加班打车回家四十一元
```
