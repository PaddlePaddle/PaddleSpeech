(简体中文|[English](./README.md)

# 定制化语音识别演示
## 介绍
定制化的语音识别是满足一些特定场景的语句识别的技术。

可以参见简单的教程：
https://aistudio.baidu.com/aistudio/projectdetail/3986429

这个 demo 是打车报销单的场景识别，定制化了地点。

## 使用方法
### 1. 配置环境
请通过 setup_docker.sh 安装镜像。进入镜像后，安装tmux (apt-get install tmux)，方便后续演示。

### 2. 演示
* bash websocket_server.sh, 完成相关资源和库的下载。这时候服务已经启动。
* 在镜像另一个终端中，bash websocket_client.sh， 通过client发送数据，得到结果。
