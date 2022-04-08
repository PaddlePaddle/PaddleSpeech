# PaddleSpeech 命令行工具

(简体中文|[English](./README.md))

`paddlespeech.cli` 模块是 PaddleSpeech 的命令行工具，它提供了最简便的方式调用 PaddleSpeech 提供的不同语音应用场景的预训练模型，用一行命令就可以进行模型预测：

 ## 命令行使用帮助
 ```bash
 paddlespeech help
 ```

 ## 声音分类
 ```bash
 paddlespeech cls --input input.wav
 ```

  ## 声纹识别

 ```bash
 paddlespeech vector --task spk --input input_16k.wav
 ```

 ## 语音识别
 ```
 paddlespeech asr --lang zh --input input_16k.wav
 ```
 
 ## 语音翻译（英-中）
 
 (暂不支持Windows系统)
 ```bash
 paddlespeech st --input input_16k.wav
 ```
 
 ## 语音合成
 ```bash
 paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```
 
 ## 文本后处理

- 标点恢复
  ```bash
  paddlespeech text --task punc --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭
  ```
