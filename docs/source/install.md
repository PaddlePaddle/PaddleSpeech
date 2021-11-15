# Installation

To avoid the trouble of environment setup, [running in Docker container](#running-in-docker-container) is highly recommended. Otherwise follow the guidelines below to install the dependencies manually.

## Prerequisites
- Python >= 3.7
- PaddlePaddle latest version (please refer to the [Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))

## Simple Setup

For user who working on `Ubuntu` with `root`  privilege.

```python
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
pip install -e .
```

For user who only needs the basic function of paddlespeech, using conda to do installing is recommended.
You can go to [minicoda](https://docs.conda.io/en/latest/miniconda.html) to select a version and install it by yourself, or you can use the scripts below to install the last miniconda version.

```python
pushd tools
bash extras/install_miniconda.sh
popd
bash
```

After installing the conda, run the setup.sh to complete the installing process.
```python
bash setup.sh
```


## Setup (Other Platform)

- Make sure these libraries or tools in [dependencies](./dependencies.md) installed. More information please see: `setup.py `and ` tools/Makefile`.
- The version of `swig` should >= 3.0
- we will do more to simplify the install process.

## Running in Docker Container (optional)

Docker is an open source tool to build, ship, and run distributed applications in an isolated environment. A Docker image for this project has been provided in [hub.docker.com](https://hub.docker.com) with all the dependencies installed. This Docker image requires the support of NVIDIA GPU, so please make sure its availiability and the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) has been installed.

Take several steps to launch the Docker image:

- Download the Docker image

For example, pull paddle 2.0.0 image:

```bash
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.0.0-gpu-cuda10.1-cudnn7
```

- Clone this repository

```
git clone https://github.com/PaddlePaddle/DeepSpeech.git
```

- Run the Docker image

```bash
sudo nvidia-docker run --rm -it -v $(pwd)/DeepSpeech:/DeepSpeech registry.baidubce.com/paddlepaddle/paddle:2.0.0-gpu-cuda10.1-cudnn7 /bin/bash
```

Now you can execute training, inference and hyper-parameters tuning in the Docker container.


- Install PaddlePaddle

For example, for CUDA 10.1, CuDNN7.5 install paddle 2.0.0:

```bash
python3 -m pip install paddlepaddle-gpu==2.0.0
```

- Install Deepspeech

Please see [Setup](#setup)  section.
