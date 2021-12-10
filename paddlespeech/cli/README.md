# PaddleSpeech Command Line

 The simplest approach to use PaddleSpeech models.

 ## Help
 ```bash
 paddlespeech help
 ```
 ## Audio Classification
 ```bash
 paddlespeech cls --input input.wav
 ```

 ## Automatic Speech Recognition
 ```
 paddlespeech asr --lang zh --input input_16k.wav
 ```
 
 ## Speech Translation (English to Chinese)
 ```bash
 paddlespeech st --input input_16k.wav
 ```
 
 ## Text-to-Speech
 ```bash
 paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
 ```
 
