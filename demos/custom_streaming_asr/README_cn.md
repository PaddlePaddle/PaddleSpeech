(简体中文|[English](./README.md))

# 定制化语音识别演示
## 介绍
在一些场景中，识别系统需要高精度的识别一些稀有词，例如导航软件中地名识别。而通过定制化识别可以满足这一需求。  

这个 demo 是打车报销单的场景识别，需要识别一些稀有的地名，可以通过如下操作实现。

* G with slot: 打车到 "address_slot"。  
![](https://ai-studio-static-online.cdn.bcebos.com/28d9ef132a7f47a895a65ae9e5c4f55b8f472c9f3dd24be8a2e66e0b88b173a4)

* 这是address slot wfst, 可以添加一些需要识别的地名.  
![](https://ai-studio-static-online.cdn.bcebos.com/47c89100ef8c465bac733605ffc53d76abefba33d62f4d818d351f8cea3c8fe2)

* 通过replace 操作, G = fstreplace(G_with_slot, address_slot), 最终可以得到定制化的解码图。  
![](https://ai-studio-static-online.cdn.bcebos.com/60a3095293044f10b73039ab10c7950d139a6717580a44a3ba878c6e74de402b)  

## 使用方法
### 1. 配置环境
安装paddle:2.2.2 docker镜像。
```
sudo docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.2

sudo docker run --privileged  --net=host --ipc=host -it --rm -v $PWD:/paddle --name=paddle_demo_docker registry.baidubce.com/paddlepaddle/paddle:2.2.2 /bin/bash 
```

### 2. 演示
* 运行如下命令，完成相关资源和库的下载和服务启动。
```
cd /paddle
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
