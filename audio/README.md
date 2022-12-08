# PaddleAudio

安装方式： pip install paddleaudio

目前支持的平台：Linux：

## Environment

## Build wheel

Linux test build whl environment:
* docker - `registry.baidubce.com/paddlepaddle/paddle:2.2.2`
* os - Ubuntu 16.04.7 LTS
* gcc/g++/gfortran - 8.2.0
* cmake - 3.18.0 (need install)

* [How to Install Docker](https://docs.docker.com/engine/install/)
* [A Docker Tutorial for Beginners](https://docker-curriculum.com/)

1. First to launch docker container.

```
docker run --privileged  --net=host --ipc=host -it --rm -v $PWD:/workspace --name=dev registry.baidubce.com/paddlepaddle/paddle:2.2.2 /bin/bash
```
2. python setup.py bdist_wheel

MAC：test build whl envrioment：
* os 
* gcc/g++/gfortran 12.2.0
* cpu Intel Xeon E5 x86_64


Windows：
not support： paddleaudio C++ extension lib (sox io, kaldi native fbank)
python setup.py bdist_wheel