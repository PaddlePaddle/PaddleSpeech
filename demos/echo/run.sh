#!/bin/bash

mkdir -p data

wav_en=data/en.wav
wav_zh=data/zh.wav

test -e ${wav_en}  || wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav -P data
test -e ${wav_zh}  || wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav -P data

pip install paddlehub

asr_en_cmd="import paddlehub as hub; model = hub.Module(name='u2_conformer_librispeech'); print(model.speech_recognize(\"${wav_en}\", device='gpu'))"
asr_zh_cmd="import paddlehub as hub; model = hub.Module(name='u2_conformer_aishell'); print(model.speech_recognize(\"${wav_zh}\", device='gpu'))"

python -c "${asr_en_cmd}"
python -c "${asr_zh_cmd}"