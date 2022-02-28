#!/bin/bash

# single process
paddlespeech tts --input 今天的天气不错啊

# Batch process
echo -e "1 欢迎光临。\n2 谢谢惠顾。" | paddlespeech tts