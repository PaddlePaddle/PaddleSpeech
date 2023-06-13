
## Environment

We develop under:
* python - >=3.8
* docker - `registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7`
* os - Ubuntu 16.04.7 LTS
* gcc/g++/gfortran - 8.2.0
* cmake - 3.16.0

> Please use `tools/env.sh` to create python `venv`, then `source venv/bin/activate` to build engine.

> We make sure all things work fun under docker, and recommend using it to develop and deploy.

* [How to Install Docker](https://docs.docker.com/engine/install/)
* [A Docker Tutorial for Beginners](https://docker-curriculum.com/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)

## Build

1. First to launch docker container.

```
docker run --privileged  --net=host --ipc=host -it --rm -v /path/to/paddlespeech:/workspace --name=dev registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7 /bin/bash
```

* More `Paddle` docker images you can see [here](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html).

2. Create python environment.

```
bash tools/venv.sh
```

2. Build `engine` and `examples`.

For now we are using feature under `develop` branch of paddle, so we need to install `paddlepaddle` nightly build version.
For example: 
```
source venv/bin/activate
python -m pip install paddlepaddle==2.4.2 -i https://mirror.baidu.com/pypi/simple
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

## FAQ

1. No moudle named `paddle`. 

```
CMake Error at CMakeLists.txt:119 (string):
  string sub-command STRIP requires two arguments.


Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'paddle'
-- PADDLE_COMPILE_FLAGS=
CMake Error at CMakeLists.txt:131 (string):
  string sub-command STRIP requires two arguments.


  File "<string>", line 1
    import os; import paddle; include_dir=paddle.sysconfig.get_include(); paddle_dir=os.path.split(include_dir)[0]; libs_dir=os.path.join(paddle_dir, 'libs'); fluid_dir=os.path.join(paddle_dir, 'fluid'); out=':'.join([libs_dir, fluid_dir]); print(out);     
    ^
```

please install paddlepaddle >= 2.4rc

2. `u2_recognizer_main: error while loading shared libraries: liblibpaddle.so: cannot open shared object file: No such file or directory`


```
cd $YOUR_ENV_PATH/lib/python3.8/site-packages/paddle/fluid
patchelf --set-soname libpaddle.so libpaddle.so
```

3. `u2_recognizer_main: error while loading shared libraries: libgfortran.so.5: cannot open shared object file: No such file or directory`

```
# my gcc version is 8.2
apt-get install gfortran-8
```

4. `Undefined reference to '_gfortran_concat_string'`

using gcc 8.2, gfortran 8.2.

5. `./boost/python/detail/wrap_python.hpp:57:11: fatal error: pyconfig.h: No such file or directory`

```
apt-get install python3-dev
```

for more info please see [here](https://github.com/okfn/piati/issues/65).
