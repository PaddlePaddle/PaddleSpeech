([简体中文](./README_cn.md)|English)

# Customized Auto Speech Recognition

## introduction
In some cases, we need to recognize the specific rare words with high accuracy. eg: address recognition in navigation apps. customized ASR can slove those issues.

this demo is customized for expense account, which need to recognize rare address.

* G with slot: 打车到 "address_slot"。
![](https://ai-studio-static-online.cdn.bcebos.com/28d9ef132a7f47a895a65ae9e5c4f55b8f472c9f3dd24be8a2e66e0b88b173a4)

* this is address slot wfst, you can add the address which want to recognize.
![](https://ai-studio-static-online.cdn.bcebos.com/47c89100ef8c465bac733605ffc53d76abefba33d62f4d818d351f8cea3c8fe2)

* after replace operation, G = fstreplace(G_with_slot, address_slot), we will get the customized graph.
![](https://ai-studio-static-online.cdn.bcebos.com/60a3095293044f10b73039ab10c7950d139a6717580a44a3ba878c6e74de402b)  

## Usage
### 1. Installation
install paddle:2.2.2 docker.
```
sudo docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.2

sudo docker run --privileged  --net=host --ipc=host -it --rm -v $PWD:/paddle --name=paddle_demo_docker registry.baidubce.com/paddlepaddle/paddle:2.2.2 /bin/bash 
```

### 2. demo
* run websocket_server.sh.  This script will download resources and libs, and launch the service.
```
bash websocket_server.sh
```
this script run in two steps:  
1. download the resources.tar.gz, those direcotries will be found in resource directory.  
model: acustic model
graph: the decoder graph (TLG.fst)  
lib: some libs  
bin: binary  
data: audio and wav.scp

2. websocket_server_main launch the service.  
some params:  
port: the service port  
graph_path: the decoder graph path  
model_path: acustic model path  
please refer other params in those files:  
PaddleSpeech/speechx/speechx/decoder/param.h  
PaddleSpeech/speechx/examples/ds2_ol/websocket/websocket_server_main.cc  

* In other terminal, run script websocket_client.sh, the client will send data and get the results.
```
bash websocket_client.sh
```
websocket_client_main will launch the client, the wav_scp is the wav set, port is the server service port.

* result:
In the log of client, you will see the message below:
```
0513 10:58:13.827821 41768 recognizer_test_main.cc:56] wav len (sample): 70208
I0513 10:58:13.884493 41768 feature_cache.h:52] set finished
I0513 10:58:24.247171 41768 paddle_nnet.h:76] Tensor neml: 10240
I0513 10:58:24.247249 41768 paddle_nnet.h:76] Tensor neml: 10240
LOG ([5.5.544~2-f21d7]:main():decoder/recognizer_test_main.cc:90)  the result of case_10 is 五月十二日二十二点三十六分加班打车回家四十一元
```