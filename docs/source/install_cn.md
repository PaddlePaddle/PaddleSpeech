(简体中文|[English](./install.md))
# 安装方法
`PaddleSpeech` 有三种安装方法。根据安装的难易程度，这三种方法可以分为 **简单**, **中等** 和 **困难**.
| 方式 | 功能                                                         | 支持系统            |
| :--- | :----------------------------------------------------------- | :------------------ |
| 简单 | (1) 使用 PaddleSpeech 的命令行功能. <br> (2) 在 Aistudio上体验 PaddleSpeech. | Linux, Mac(不支持M1芯片)，Windows |
| 中等 | 支持 PaddleSpeech 主要功能，比如使用已有 examples 中的模型和使用 PaddleSpeech 来训练自己的模型. | Linux               |
| 困难 | 支持 PaddleSpeech 的各项功能，包含训练语言模型,使用强制对齐等。并且你更能成为一名开发者！ | Ubuntu              |
## 先决条件
- Python >= 3.7
- 最新版本的 PaddlePaddle (请看 [安装向导](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))
- C++ 编译环境
- 提示: 对于 Linux 和 Mac，请不要使用 `sh` 代替安装文档中的 `bash`
## 简单： 获取基本功能(支持 Linux，Mac 和 Windows)
- 如果你是一个刚刚接触 `PaddleSpeech` 的新人并且想要很方便地体验一下该项目。我们建议你 体验一下[AI Studio](https://aistudio.baidu.com/aistudio/index)。我们在AI Studio上面建立了一个让你一步一步运行体验来使用`PaddleSpeech`的教程。
- 如果你想使用 `PaddleSpeech` 的命令行功能，你需要跟随下面的步骤来安装 `PaddleSpeech`。如果你想了解更多关于使用 `PaddleSpeech` 命令行功能的信息，你可以参考 [cli](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/cli)。
### 安装Conda
Conda是一个包管理的环境。你可以前往 [minicoda](https://docs.conda.io/en/latest/miniconda.html) 去下载并安装 conda（请下载 py>=3.7 的版本）。
然后你需要安装 `paddlespeech` 的 conda 依赖:
```bash
conda install -y -c conda-forge sox libsndfile bzip2
```
### 安装 C++ 编译环境
(如果你系统上已经安装了 C++ 编译环境，请忽略这一步。)
#### Windows
对于 Windows 系统，需要安装 `Visual Studio` 来完成 C++ 编译环境的安装。
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
### 安装 PaddleSpeech
你可以使用如下命令：
```bash
pip install paddlepaddle paddlespeech
```
## 中等： 获取主要功能（支持 Linux）
如果你想要使用` paddlespeech` 的主要功能。你需要完成以下几个步骤
### Git clone PaddleSpeech
你需要先git clone本仓库
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
```

### 安装 Conda
Conda 是一个包管理的环境。你可以前往 [minicoda](https://docs.conda.io/en/latest/miniconda.html) 去下载并安装 conda（请下载 py>=3.7 的版本）。你可以尝试自己安装，或者使用以下的命令：
```bash
# 下载 miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
# 安装 miniconda
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
# conda 初始化
$HOME/miniconda3/bin/conda init
# 激活 conda
bash
```
然后你可以创建一个 conda 的虚拟环境：
```bash
conda create -y -p tools/venv python=3.7
```
激活 conda 虚拟环境：
```bash
conda activate tools/venv
```
安装 `paddlespeech` 的 conda 依赖：
```bash
conda install -y -c conda-forge sox libsndfile swig bzip2
```
### 安装 C++ 编译环境
(如果你系统上已经安装了 C++ 编译环境，请忽略这一步。)
你可以使用如下的步骤来安装 C++ 的编译环境 `gcc`  和 `gxx`：
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
（提示： 如果你想使用**困难**方式完成安装，请不要使用最后一条命令）
### 安装 PaddlePaddle
你可以根据系统配置选择 PaddlePaddle 版本，例如系统使用 CUDA 10.2， CuDNN7.5 ，你可以安装 paddlepaddle-gpu 2.2.0：
```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```
### 安装 PaddleSpeech
最后安装 `paddlespeech`，这样你就可以使用 `paddlespeech`中已有的 examples：
```bash
pip install .
```
## 困难： 获取所有功能（支持 Ubuntu）
### 先决条件
- Ubuntu >= 16.04
- 选择 1： 使用`Ubuntu` docker。
- 选择 2： 使用`Ubuntu` ，并且拥有 root 权限。

为了避免各种环境配置问题，我们非常推荐你使用 docker 容器。如果你不想使用 docker，但是可以使用拥有 root 权限的 Ubuntu 系统，你也可以完成**困难**方式的安装。
### 选择1： 使用Docker容器（推荐）
Docker 是一种开源工具，用于在和系统本身环境相隔离的环境中构建、发布和运行各类应用程序。你可以访问 [hub.docker.com](https://hub.docker.com) 来下载各种版本的 docker，目前已经有适用于 `PaddleSpeech` 的 docker 提供在了该网站上。Docker 镜像需要使用 Nvidia GPU，所以你也需要提前安装好 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 。
你需要完成几个步骤来启动docker：
- 下载 docker 镜像:
  例如，拉取 paddle2.2.0 镜像：
```bash
sudo nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7
```
- 克隆 `PaddleSpeech` 仓库
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
```
- 启动 docker 镜像
```bash
sudo nvidia-docker run --net=host --ipc=host --rm -it -v $(pwd)/PaddleSpeech:/PaddleSpeech registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash
```
- 进入 PaddleSpeech 目录
```bash
cd /PaddleSpeech
```
完成这些以后，你就可以在 docker 容器中执行训练、推理和超参 fine-tune。
### 选择2： 使用有 root 权限的 Ubuntu
- 使用apt安装 `build-essential`
```bash
sudo apt install build-essential
```
- 克隆 `PaddleSpeech` 仓库
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
# 进入PaddleSpeech目录
cd PaddleSpeech
```
### 安装 Conda
```bash
# 下载 miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P tools/
# 安装 miniconda
bash tools/Miniconda3-latest-Linux-x86_64.sh -b
# conda 初始化
$HOME/miniconda3/bin/conda init
# 激活 conda
bash
# 创建 Conda 虚拟环境
conda create -y -p tools/venv python=3.7
# 激活 Conda 虚拟环境:
conda activate tools/venv
# 安装 Conda 包
conda install -y -c conda-forge sox libsndfile swig bzip2 libflac bc
```
### 安装 PaddlePaddle
请确认你系统是否有 GPU，并且使用了正确版本的 paddlepaddle。例如系统使用 CUDA 10.2, CuDNN7.5 ，你可以安装 paddlepaddle-gpu 2.2.0：
```bash
python3 -m pip install paddlepaddle-gpu==2.2.0
```
### 用开发者模式安装 PaddleSpeech
```bash
pip install -e .[develop]
```
### 安装 Kaldi（可选）
```bash
pushd tools
bash extras/install_openblas.sh
bash extras/install_kaldi.sh
popd
```
