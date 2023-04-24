# PaddleAudio

安装方式： pip install paddleaudio

目前支持的平台：Linux, Mac, Windows

## Environment

## Build wheel
cmd: python setup.py bdist_wheel

Linux test build whl environment:
* os - Ubuntu 16.04.7 LTS
* gcc/g++ - 8.2.0
* cmake - 3.18.0 (need install)

MAC：test build whl envrioment：
* os 
* gcc/g++ 12.2.0
* cpu Intel Xeon E5 x86_64

Windows：
not support paddleaudio C++ extension lib (sox io, kaldi native fbank)
