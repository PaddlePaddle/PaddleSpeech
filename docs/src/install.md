# Installation

To avoid the trouble of environment setup, [running in Docker container](#running-in-docker-container) is highly recommended. Otherwise follow the guidelines below to install the dependencies manually.

## Prerequisites
- Python >= 3.7
- PaddlePaddle 2.0.0 or later (please refer to the [Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))

## Setup

- Make sure these libraries or tools installed: `pkg-config`, `flac`, `ogg`, `vorbis`, `boost`, `sox, and `swig`, e.g. installing them via `apt-get`:

```bash
sudo apt-get install -y sox pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev
```

or, installing them via `yum`:

```bash
sudo yum install pkgconfig libogg-devel libvorbis-devel boost-devel python3-devel
wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.1.tar.xz
xz -d flac-1.3.1.tar.xz
tar -xvf flac-1.3.1.tar
cd flac-1.3.1
./configure
make
make install
```

- Run the setup script for the remaining dependencies

```bash
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
pushd tools; make; popd
source tools/venv/bin/activate
bash setup.sh
```

- Source venv before do experiment.

```bash
source tools/venv/bin/activate
```

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
