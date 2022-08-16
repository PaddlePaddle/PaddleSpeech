#!/bin/bash

rm -rf *.wav
./run_sedit_en_new.sh       # 语音编辑任务(英文)
./run_gen_en_new.sh         # 个性化语音合成任务(英文)
./run_clone_en_to_zh_new.sh # 跨语言语音合成任务(英文到中文的语音克隆)