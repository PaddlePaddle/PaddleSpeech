# Installation

There are 3 ways to use the repository. According to the degree of difficulty, the 3 ways can be divided into Easy, Medium and Hard.



## Easy: Get the Basic Funcition Without Your Own Mechine

If you are in touch with PaddleSpeech for the first time and want to experience it easily without your own mechine. We recommand you to go to aistudio to experience the PaddleSpeech project. There is a step-by-step tutorial for PaddleSpeech and you can use the basic function of PaddleSpeech with a free machine.



## Prerequisites for Medium and Hard

- Python >= 3.7
- PaddlePaddle latest version (please refer to the [Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))
- Only Linux is supported
- Hip: Do not use command `sh` instead of command `bash`



## Medium: Get the Basic Funciton on Your Mechine

If you want to install the paddlespeech on your own mechine. There are 3 steps you need to do.

### Install PaddlePaddle

```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```

### Install the Conda

The first setup is installing the conda. Conda is environment management system. You can go to [minicoda](https://docs.conda.io/en/latest/miniconda.html) to select a version (py>=3.7) and install it by yourself or you can use the scripts below:

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

Then you can create an conda virtual environment using the script:

```bash
conda create -n py37 python=3.7
```

Activate the conda virtual environment:

```bash
conda activate py37
```

Intall the conda dependencies

```bash
conda install -c conda-forge sox libsndfile swig bzip2 gcc_linux-64=8.4.0 gxx_linux-64=8.4.0 --yes
```

### Install the PaddleSpeech Using PiP

To Install the PaddleSpeech, there are two methods. You can use the script below:

```bash
pip install paddlespeech
```

If you install the paddlespeech by pip, you can use it to help you to build your own model. However, you can not use the ready-made examples in paddlespeech. 

If you want to use the ready-made examples in paddlespeech, you need to clone the repository and install the paddlespeech package.

```bash
https://github.com/PaddlePaddle/PaddleSpeech.git
## Into the PaddleSpeech
cd PaddleSpeech
pip install .
```



## Hard: Get the Full Funciton on Your Mechine

### Prerequisites

- choice 1: working with `ubuntu` Docker Container.

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

```
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
```

- Run the Docker image

```bash
sudo nvidia-docker run --rm -it -v $(pwd)/PaddleSpeech:/PaddleSpeech registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
```

Now you can execute training, inference and hyper-parameters tuning in the Docker container.


- Install PaddlePaddle

For example, for CUDA 10.2, CuDNN7.5 install paddle 2.2.0:

```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```


### Choice 2: Running in Ubuntu with Root Privilege

- Clone this repository

```
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
conda create -n py37 python=3.7
# Activate the conda virtual environment:
conda activate py37
# Install the conda packags
conda install -c conda-forge sox libsndfile swig bzip2 gcc_linux-64=8.4.0 gxx_linux-64=8.4.0 --yes
```

### Get the Funcition for Developing PaddleSpeech

```bash
pip install .[develop]
```

### Install the Kaldi

```bash
pushd tools
bash extras/install_openblas.sh
bash extras/install_kaldi.sh
popd
```




## Setup for Other Platform 

- Make sure these libraries or tools in [dependencies](./dependencies.md) installed. More information please see: `setup.py `and ` tools/Makefile`.
- The version of `swig` should >= 3.0
- we will do more to simplify the install process.
- Install Paddlespeech