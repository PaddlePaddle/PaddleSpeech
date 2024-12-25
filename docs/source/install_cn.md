(简体中文|[English](./install.md))
# 安装方法
`PaddleSpeech` 有三种安装方法。根据安装的难易程度，这三种方法可以分为 **简单**, **中等** 和 **困难**.
| 方式 | 功能                                                         | 支持系统            |
| :--- | :----------------------------------------------------------- | :------------------ |
| 简单 | (1) 使用 PaddleSpeech 的命令行功能. <br> (2) 在 Aistudio上体验 PaddleSpeech. | Linux, Mac(不支持M1芯片)，Windows (安装详情查看[#1195](https://github.com/PaddlePaddle/PaddleSpeech/discussions/1195)) |
| 中等 | 支持 PaddleSpeech 主要功能，比如使用已有 examples 中的模型和使用 PaddleSpeech 来训练自己的模型. | Linux, Mac(不支持M1芯片，不支持训练), Windows(不支持训练)               |
| 困难 | 支持 PaddleSpeech 的各项功能，包含结合 kaldi 使用 join ctc decoder 方式解码 ([asr2](../../examples/librispeech/asr2 ))，训练语言模型,使用强制对齐等。并且你更能成为一名开发者！ | Ubuntu              |
## 先决条件
- Python >= 3.7
- 最新版本的 PaddlePaddle (请看 [安装向导](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html))
- C++ 编译环境
- 提示: 对于 Linux 和 Mac，请不要使用 `sh` 代替安装文档中的 `bash`
- 提示: 我们建议在安装 `paddlepaddle` 的时候使用百度源 https://mirror.baidu.com/pypi/simple ，而在安装 `paddlespeech` 的时候使用清华源 https://pypi.tuna.tsinghua.edu.cn/simple 。

## 简单： 获取基本功能(支持 Linux，Mac 和 Windows)
- 如果你是一个刚刚接触 `PaddleSpeech` 的新人并且想要很方便地体验一下该项目。我们建议你体验一下 [AI Studio](https://aistudio.baidu.com/aistudio/index)。我们在 AI Studio上面建立了一个让你一步一步运行体验来使用 `PaddleSpeech` 的[教程](https://aistudio.baidu.com/aistudio/education/group/info/25130)。
- 如果你想使用 `PaddleSpeech` 的命令行功能，你需要跟随下面的步骤来安装 `PaddleSpeech`。如果你想了解更多关于使用 `PaddleSpeech` 命令行功能的信息，你可以参考 [cli](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/cli)。
### 安装 Conda
Conda是一个包管理的环境。你可以前往 [miniconda](https://docs.conda.io/en/latest/miniconda.html) 去下载并安装 conda（请下载 py>=3.7 的版本）。
然后你需要安装 `paddlespeech` 的 conda 依赖:
```bash
conda install -y -c conda-forge sox libsndfile bzip2
```
### 安装 C++ 编译环境
(如果你系统上已经安装了 C++ 编译环境，请忽略这一步。)
#### Windows
对于 Windows 系统，需要安装 `Visual Studio` 来完成 C++ 编译环境的安装。

https://visualstudio.microsoft.com/visual-cpp-build-tools/

你可以前往讨论区[#1195](https://github.com/PaddlePaddle/PaddleSpeech/discussions/1195)获取更多帮助。

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
部分用户系统由于默认源的问题，安装中会出现kaldiio安转出错的问题，建议首先安装pytest-runner:
```bash
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple 
```
然后你可以使用如下命令：
```bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple
```
你也可以安装指定版本的paddlepaddle，或者安装 develop 版本。
```bash
# 安装2.3.1版本. 注意：2.3.1只是一个示例，请按照对paddlepaddle的最小依赖进行选择。
pip install paddlepaddle==2.3.1 -i https://mirror.baidu.com/pypi/simple
# 安装 develop 版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```
> 如果您在使用 paddlespeech 的过程中遇到关于下载 **nltk_data** 的问题，可能是您的网络不佳，我们建议您下载我们提供的 [nltk_data](https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz) 并解压缩到您的 `${HOME}` 目录下。

> 如果出现 paddlespeech-ctcdecoders 无法安装的问题，无须担心，这个只影响 deepspeech2 模型的推理，不影响其他模型的使用。

## 中等： 获取主要功能（支持 Linux， Mac 和 Windows 不支持训练）
如果你想要使用 `paddlespeech` 的主要功能。你需要完成以下几个步骤
### Git clone PaddleSpeech
你需要先 git clone 本仓库
```bash
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
```
### 安装 Conda
Conda 是一个包管理的环境。你可以前往 [minicoda](https://docs.conda.io/en/latest/miniconda.html) 去下载并安装 conda（请下载 py>=3.7 的版本）。windows 系统可以使用 conda 的向导安装，linux 和 mac 可以使用以下的命令：
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
conda create -y -p tools/venv python=3.8
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
你可以根据系统配置选择 PaddlePaddle 版本，例如系统使用 CUDA 10.2， CuDNN7.6，你可以安装 paddlepaddle-gpu 2.4.1：
```bash
# 注意：2.4.1 只是一个示例，请按照对paddlepaddle的最小依赖进行选择。
python3 -m pip install paddlepaddle-gpu==2.4.1 -i https://mirror.baidu.com/pypi/simple
```
你也可以安装 develop 版本的PaddlePaddle. 例如系统使用 CUDA 10.2， CuDNN7.6 ，你可以安装 paddlepaddle-gpu develop:
```bash
python3 -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### 安装 PaddleSpeech
最后安装 `paddlespeech`，这样你就可以使用 `paddlespeech` 中已有的 examples：
```bash
# 部分用户系统由于默认源的问题，安装中会出现 kaldiio 安转出错的问题，建议首先安装pytest-runner:
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple 
# 请确保目前处于PaddleSpeech项目的根目录
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 困难： 获取所有功能（支持 Ubuntu）
### 先决条件
- Ubuntu >= 16.04
- 选择 1： 使用`Ubuntu` docker。
- 选择 2： 使用`Ubuntu` ，并且拥有 root 权限。

为了避免各种环境配置问题，我们非常推荐你使用 docker 容器。如果你不想使用 docker，但是可以使用拥有 root 权限的 Ubuntu 系统，你也可以完成**困难**方式的安装。
### 选择1： 使用 Docker 容器（推荐）
Docker 是一种开源工具，用于在和系统本身环境相隔离的环境中构建、发布和运行各类应用程序。如果您没有 Docker 运行环境，请参考 [Docker 官网](https://www.docker.com/)进行安装，如果您准备使用 GPU 版本镜像，还需要提前安装好 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 。 

我们提供了包含最新 PaddleSpeech 代码的 docker 镜像，并预先安装好了所有的环境和库依赖，您只需要**拉取并运行 docker 镜像**，无需其他任何额外操作，即可开始享用 PaddleSpeech 的所有功能。

在 [Docker Hub](https://hub.docker.com/repository/docker/paddlecloud/paddlespeech) 中获取这些镜像及相应的使用指南，包括 CPU、GPU、ROCm 版本。

如果您对自动化制作 docker 镜像感兴趣，或有自定义需求，请访问 [PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton) 做进一步了解。
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
conda create -y -p tools/venv python=3.8
# 激活 Conda 虚拟环境:
conda activate tools/venv
# 安装 Conda 包
conda install -y -c conda-forge sox libsndfile swig bzip2 libflac bc
```
### 安装 PaddlePaddle
请确认你系统是否有 GPU，并且使用了正确版本的 paddlepaddle。例如系统使用 CUDA 10.2, CuDNN7.6 ，你可以安装 paddlepaddle-gpu 2.4.1：
```bash
# 注意：2.4.1 只是一个示例，请按照对paddlepaddle的最小依赖进行选择。
python3 -m pip install paddlepaddle-gpu==2.4.1 -i https://mirror.baidu.com/pypi/simple
```
你也可以安装 develop 版本的PaddlePaddle. 例如系统使用 CUDA 10.2， CuDNN7.6 ，你可以安装 paddlepaddle-gpu develop:
```bash
python3 -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### 用开发者模式安装 PaddleSpeech
部分用户系统由于默认源的问题，安装中会出现 kaldiio 安转出错的问题，建议首先安装 pytest-runner:
```bash
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple 
```
然后安装 PaddleSpeech：
```bash
pip install -e .[develop] -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 安装 Kaldi（可选）
```bash
pushd tools
bash extras/install_openblas.sh
bash extras/install_kaldi.sh
popd
```
