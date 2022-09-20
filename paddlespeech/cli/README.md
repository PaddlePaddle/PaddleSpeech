# PaddleSpeech Command Line

([简体中文](./README_cn.md)|English)

 The simplest approach to use PaddleSpeech models.

 ## Help
 ```bash
 paddlespeech help
 ```
 ## Audio Classification
 ```bash
 paddlespeech cls --input input.wav
 ```

 ## Speaker Verification

 ```bash
 paddlespeech vector --task spk --input input_16k.wav
 ```

 ## Automatic Speech Recognition
 ```
 paddlespeech asr --lang zh --input input_16k.wav
 ```
 
 ## Speech Translation (English to Chinese)
 
 (not support for Windows now)
 ```bash
 paddlespeech st --input input_16k.wav
 ```
 
 ## Text-to-Speech
 ```bash
 paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```
 
 ## Text Post-precessing

- Punctuation Restoration
  ```bash
  paddlespeech text --task punc --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭
  ```
- Faster Punctuation Restoration
  ```bash
  paddlespeech text --task punc --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭 --model ernie_linear_p3_wudao_fast
  ```
