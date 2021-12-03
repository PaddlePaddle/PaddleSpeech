# Installation
There are 3 ways to use `PaddleSpeech`. According to the degree of difficulty, the 3 ways can be divided into `Easy`, `Medium` and `Hard`.

## Easy: Get the Basic Funcition Without Your Own Mechine
If you are a newer of `PaddleSpeech` and want to experience it easily without your own mechine. We recommand you to use [AI Studio] (https://aistudio.baidu.com/aistudio/index) to experience it. There is a step-by-step tutorial for `PaddleSpeech` and you can use the basic function of `PaddleSpeech` with a free machine.

## Prerequisites for Medium and Hard
- Python >= 3.7
- PaddlePaddle latest version (please refer to the [Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))
- Only Linux is supported
- Hip: Do not use command `sh` instead of command `bash`

## Medium: Get the Basic Funciton on Your Mechine
If you want to install `paddlespeech` on your own mechine. There are 3 steps you need to do.

### Install the Conda
Conda is environment management system. You can go to [minicoda](https://docs.conda.io/en/latest/miniconda.html) to select a version (py>=3.7) and install it by yourself or you can use the following command:
```bash
# download the miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# install the miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b
# conda init
$HOME/miniconda3/bin/conda init
# activate the conda
bash
```
Then you can create an conda virtual environment using the following command:
```bash
conda create -y -p tools/venv python=3.7
```
Activate the conda virtual environment:
```bash
conda activate tools/venv
```
Install  conda dependencies for `paddlespeech` :
```bash
conda install -y -c conda-forge sox libsndfile swig bzip2 gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
```
### Install PaddlePaddle
For example, for CUDA 10.2, CuDNN7.5 install paddle 2.2.0:
```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```
### Install PaddleSpeech 
To Install  `paddlespeech`, there are two methods. You can use the following command:
```bash
pip install paddlespeech
```
If you install `paddlespeech` by `pip`, you can use it to help you build your own model. However, you can not use the `ready-made `examples in paddlespeech. 

If you want to use the` ready-made `examples in `paddlespeech`, you need to clone this repository and install  `paddlespeech`  by the foll
```bash
https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
pip install .
```
## Hard: Get the Full Funciton on Your Mechine
### Prerequisites
- choice 1: working with `Ubuntu` Docker Container.

  or

- choice 2: working on `Ubuntu` with `root` privilege. 

To avoid the trouble of environment setup, [running in Docker container](#running-in-docker-container) is highly recommended. Otherwise If you work on `Ubuntu` with `root` privilege, you can skip the next step.

### Choice 1: Running in Docker Container (Recommand)
Docker is an open source tool to build, ship, and run distributed applications in an isolated environment. A Docker image for this project has been provided in [hub.docker.com](https://hub.docker.com) with all the dependencies installed. This Docker image requires the support of NVIDIA GPU, so please make sure its availiability and the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) has been installed.

Take several steps to launch the Docker image:
- Download the Docker image

For example, pull paddle 2.2.0 image:
```bash
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7
```
- Clone this repository
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
```
- Run the Docker image

```bash
sudo nvidia-docker run --net=host --ipc=host --rm -it -v $(pwd)/PaddleSpeech:/PaddleSpeech registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
```
Now you can execute training, inference and hyper-parameters tuning in  Docker container.
### Choice 2: Running in Ubuntu with Root Privilege
- Clone this repository
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
```
Install paddle 2.2.0:
```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```
### Install the Conda
```bash
# download and install the miniconda
pushd tools
bash extras/install_miniconda.sh
popd
# use the "bash" command to make the conda environment works
bash
# create an conda virtual environment
conda create -y -n tools/venv python=3.7
# Activate the conda virtual environment:
conda activate tools/venv
# Install the conda packags
conda install -y -c conda-forge sox libsndfile swig bzip2 libflac bc gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
```
### Install PaddlePaddle
For example, for CUDA 10.2, CuDNN7.5 install paddle 2.2.0:

```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```
### Get the Funcition for Developing PaddleSpeech
```bash
pip install .[develop]
```
### Install the Kaldi (Optional)
```bash
pushd tools
bash extras/install_openblas.sh
bash extras/install_kaldi.sh
popd
```


## Setup for Other Platform 
- Make sure these libraries or tools in [dependencies](./dependencies.md) installed. More information please see: `setup.py `and ` tools/Makefile`.
- The version of `swig` should >= 3.0
- we will simplify the install process in the future.

