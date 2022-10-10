# SpeechX -- All in One Speech Task Inference 

## Environment

We develop under:
* python - 3.7
* docker - `registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7`
* os - Ubuntu 16.04.7 LTS
* gcc/g++/gfortran - 8.2.0
* cmake - 3.16.0

> Please using `tools/env.sh` to create python `venv`, then `source venv/bin/activate` to build speechx.

> We make sure all things work fun under docker, and recommend using it to develop and deploy.

* [How to Install Docker](https://docs.docker.com/engine/install/)
* [A Docker Tutorial for Beginners](https://docker-curriculum.com/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

## Build

1. First to launch docker container.

```
docker run --privileged  --net=host --ipc=host -it --rm -v $PWD:/workspace --name=dev registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7 /bin/bash
```

* More `Paddle` docker images you can see [here](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html).

2. Create python environment.

```
bash tools/venv.sh
```

2. Build `speechx` and `examples`.

```
source venv/bin/activate
./build.sh
```

3. Go to `examples` to have a fun.

More details please see `README.md` under `examples`.


## Valgrind (Optional)

> If using docker please check `--privileged` is set when `docker run`.

* Fatal error at startup: `a function redirection which is mandatory for this platform-tool combination cannot be set up`
```bash
apt-get install libc6-dbg
```

* Install

```bash
pushd tools
./setup_valgrind.sh
popd
```

## TODO

### Deepspeech2 with linear feature
* DecibelNormalizer: there is a small difference between the offline and online db norm. The computation of online db norm reads features chunk by chunk, which causes the feature size to be different different with offline db norm. In `normalizer.cc:73`, the `samples.size()` is different, which causes the different result.
