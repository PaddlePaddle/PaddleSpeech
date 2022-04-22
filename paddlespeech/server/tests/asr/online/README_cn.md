([English](./README.md)|中文)

# 语音服务

## 介绍
本文档介绍如何使用流式ASR的一种不同客户端:麦克风。 


## 使用方法
### 1. 安装
请看 [安装文档](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install.md).

推荐使用 **paddlepaddle 2.2.1** 或以上版本。
你可以从 medium，hard 三中方式中选择一种方式安装 PaddleSpeech。


### 2. 准备测试文件

这个 ASR client 的输入应该是一个 WAV 文件（`.wav`），并且采样率必须与模型的采样率相同。

可以下载此 ASR client的示例音频：
```bash
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 2. 流式 ASR 客户端使用方法

- Python模拟流式服务命令行
   ```

   # 流式ASR
   paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8091 --input ./zh.wav

   ```


- 麦克风
   ```
   # 直接调用麦克风设备
   python microphone_client.py

   ```
