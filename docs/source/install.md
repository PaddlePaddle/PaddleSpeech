([简体中文](./install_cn.md)|English)
# Installation
There are 3 ways to use `PaddleSpeech`. According to the degree of difficulty, the 3 ways can be divided into **Easy**, **Medium**, and **Hard**. You can choose one of the 3 ways to install `PaddleSpeech`.

| Way | Function                                                     | Support|
|:---- |:----------------------------------------------------------- |:----|
| Easy     | (1) Use command-line functions of PaddleSpeech. <br> (2) Experience PaddleSpeech on Ai Studio. | Linux, Mac(not support M1 chip)，Windows ( For more information about installation, see [#1195](https://github.com/PaddlePaddle/PaddleSpeech/discussions/1195)) |
| Medium     | Support major functions ，such as using the` ready-made `examples and using PaddleSpeech to train your model.                                           | Linux, Mac(not support M1 chip, not support training models)，Windows (not support training models) |
| Hard     | Support full function of Paddlespeech, including using join ctc decoder with kaldi([asr2](../../examples/librispeech/asr2 )), training n-gram language model, Montreal-Forced-Aligner, and so on. And you are more able to be a developer! | Ubuntu |

## Prerequisites
- Python >= 3.7
- PaddlePaddle latest version (please refer to the [Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))
- C++ compilation environment
- Tip: For Linux and Mac, do not use command `sh` instead of command `bash` in installation document.
- Tip: We recommand you to install `paddlepaddle` from https://mirror.baidu.com/pypi/simple and install `paddlespeech` from https://pypi.tuna.tsinghua.edu.cn/simple. 

## Easy: Get the Basic Function (Support Linux, Mac, and Windows)
- If you are newer to `PaddleSpeech` and want to experience it easily without your machine. We recommend you to use [AI Studio](https://aistudio.baidu.com/aistudio/index) to experience it. There is a step-by-step [tutorial](https://aistudio.baidu.com/aistudio/education/group/info/25130) for `PaddleSpeech`, and you can use the basic function of `PaddleSpeech` with a free machine.
- If you want to use the command line function of Paddlespeech, you need to complete the following steps to install `PaddleSpeech`. For more information about how to use the command line function, you can see the [cli](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/cli).
### Install Conda
Conda is a management system of the environment. You can go to [miniconda](https://docs.conda.io/en/latest/miniconda.html)  (select a version py>=3.7) to download and install the conda.
And then Install  conda dependencies for `paddlespeech` :

```bash
conda install -y -c conda-forge sox libsndfile bzip2
```
### Install C++ Compilation Environment 
(If you already have C++ compilation environment, you can miss this step.)
#### Windows
You need to install `Visual Studio` to make the C++ compilation environment.

https://visualstudio.microsoft.com/visual-cpp-build-tools/

You can also see [#1195](https://github.com/PaddlePaddle/PaddleSpeech/discussions/1195) for more help.

#### Mac
```bash
brew install gcc
```
#### Linux
```bash
#  centos
sudo yum install gcc gcc-c++
```
```bash
# ubuntu
sudo apt install build-essential
```
```bash
# Others
conda install -y -c gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
```
### Install PaddleSpeech 
Some users may fail to install `kaldiio` due to the default download source, you can install `pytest-runner` at first；
```bash
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
```
Then you can use the following commands:
```bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple
```
You can also specify the version of paddlepaddle or install the develop version. 
```bash
# install 2.4.1 version. Note, 2.4.1 is just an example, please follow the minimum dependency of paddlepaddle for your selection
pip install paddlepaddle==2.4.1 -i https://mirror.baidu.com/pypi/simple
# install develop version
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```
> If you encounter problem with downloading **nltk_data** while using paddlespeech, it maybe due to your poor network, we suggest you download the [nltk_data](https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz) provided by us, and extract it to your `${HOME}`.

> If you fail to install paddlespeech-ctcdecoders, you only can not use deepspeech2 model inference. For other models, it doesn't matter.
 
## Medium: Get the Major Functions (Support Linux, mac and windows not support training)
If you want to get the major function of  `paddlespeech`, you need to do following steps:
### Git clone PaddleSpeech
You need to `git clone` this repository at first.
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
```

### Install Conda
Conda is a management system of the environment. You can go to [minicoda](https://docs.conda.io/en/latest/miniconda.html) to select a version (py>=3.7). For windows, you can follow the installing guide step by step and for linux and mac, you can use the following commands:
```bash
# download the miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
# install the miniconda
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
# conda init
$HOME/miniconda3/bin/conda init
# activate the conda
bash
```
Then you can create a conda virtual environment using the following command:
```bash
conda create -y -p tools/venv python=3.8
```
Activate the conda virtual environment:
```bash
conda activate tools/venv
```
Install  conda dependencies for `paddlespeech` :
```bash
conda install -y -c conda-forge sox libsndfile swig bzip2
```
### Install C++ Compilation Environment 
(If you already have C++ compilation environment, you can miss this step.)
Do not forget to install `gcc` and `gxx` on your system.
You can choose to use the scripts below to install them.

```bash
#  centos
sudo yum install gcc gcc-c++
```
```bash
# ubuntu
sudo apt install build-essential
```
```bash
# Others
conda install -y -c gcc_linux-64=8.4.0 gxx_linux-64=8.4.0
```
(Hip: Do not use the last script if you want to install by **Hard** way):
### Install PaddlePaddle
You can choose the `PaddlePaddle` version based on your system. For example, for CUDA 10.2, CuDNN7.6 install paddlepaddle-gpu 2.4.1:
```bash
# Note, 2.4.1 is just an example, please follow the minimum dependency of paddlepaddle for your selection
python3 -m pip install paddlepaddle-gpu==2.4.1 -i https://mirror.baidu.com/pypi/simple
```
You can also install the develop version of paddlepaddle. For example, for CUDA 10.2, CuDNN7.6 install paddlepaddle-gpu develop:
```bash
python3 -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### Install PaddleSpeech 
You can install  `paddlespeech`  by the following command，then you can use the `ready-made` examples in `paddlespeech` :
```bash
# Some users may fail to install `kaldiio` due to the default download source, you can install `pytest-runner` at first；
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
# Make sure you are in the root directory of PaddleSpeech
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Hard: Get the Full Function (Support Ubuntu)
### Prerequisites
- Ubuntu >= 16.04.
- choice 1: working with `Ubuntu` Docker Container.
- choice 2: working on `Ubuntu` with `root` privilege. 

To avoid the trouble of environment setup, running in a Docker container is highly recommended. Otherwise, if you work on `Ubuntu` with `root` privilege, you can still complete the installation.

### Choice 1: Running in Docker Container (Recommend)
Docker is an open-source tool to build, ship, and run distributed applications in an isolated environment. If you  do not have a Docker environment, please refer to [Docker](https://www.docker.com/). If you will use GPU version, you also need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

We provide docker images containing the latest PaddleSpeech code, and all environment and package dependencies are pre-installed. All you have to do is to **pull and run the docker image**. Then you can enjoy PaddleSpeech without any extra steps.

Get these images and guidance in [docker hub](https://hub.docker.com/repository/docker/paddlecloud/paddlespeech), including CPU, GPU, ROCm environment versions. 

If you have some customized requirements about automatic building docker images, you can get it in github repo [PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton).

### Choice 2: Running in Ubuntu with Root Privilege
- Install `build-essential` by apt
```bash
sudo apt install build-essential
```
- Clone this repository
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
# Enter the PaddleSpeech dir
cd PaddleSpeech
```
### Install the Conda
```bash
# download the miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
# install the miniconda
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
# conda init
$HOME/miniconda3/bin/conda init
# use the "bash" command to make the conda environment works
bash
# create a conda virtual environment
conda create -y -p tools/venv python=3.8
# Activate the conda virtual environment:
conda activate tools/venv
# Install the conda packages
conda install -y -c conda-forge sox libsndfile swig bzip2 libflac bc
```
### Install PaddlePaddle
Make sure you have GPU and the paddlepaddle version is right. For example, for CUDA 10.2, CuDNN7.6 install paddle 2.4.1:
```bash
# Note, 2.4 is just an example, please follow the minimum dependency of paddlepaddle for your selection
python3 -m pip install paddlepaddle-gpu==2.4.1 -i https://mirror.baidu.com/pypi/simple
```
You can also install the develop version of paddlepaddle. For example, for CUDA 10.2, CuDNN7.6 install paddlepaddle-gpu develop:
```bash
python3 -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### Install PaddleSpeech in Developing Mode
Some users may fail to install `kaldiio` due to the default download source, you can install `pytest-runner` at first:
```bash
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
```
Then install PaddleSpeech:
```bash
pip install -e .[develop] -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### Install the Kaldi (Optional)
```bash
pushd tools
bash extras/install_openblas.sh
bash extras/install_kaldi.sh
popd
```
