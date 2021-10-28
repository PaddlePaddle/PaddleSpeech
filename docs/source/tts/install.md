# Installation
## Install PaddlePaddle
Parakeet requires PaddlePaddle as its backend. Note that 2.1.2 or newer versions of paddle is required.

Since paddlepaddle has multiple packages depending on the device (cpu or gpu) and the dependency libraries, it is recommended to install a proper package of paddlepaddle with respect to the device and dependency library versons via `pip`.

Installing paddlepaddle with conda or build paddlepaddle from source is also supported. Please refer to [PaddlePaddle installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) for more details.

Example instruction to install paddlepaddle via pip is listed below.

### PaddlePaddle with GPU
```python
# PaddlePaddle for CUDA10.1 
python -m pip install paddlepaddle-gpu==2.1.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# PaddlePaddle for CUDA10.2  
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# PaddlePaddle for CUDA11.0
python -m pip install paddlepaddle-gpu==2.1.2.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# PaddlePaddle for CUDA11.2 
python -m pip install paddlepaddle-gpu==2.1.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
### PaddlePaddle with CPU
```python
python -m pip install paddlepaddle==2.1.2 -i https://mirror.baidu.com/pypi/simple
```
## Install libsndfile
Experimemts in parakeet often involve audio and spectrum processing, thus `librosa` and `soundfile` are required. `soundfile` requires a extra C library `libsndfile`, which is not always handled by pip.

For Windows and Mac users, `libsndfile` is also installed when installing `soundfile` via pip, but for Linux users, installing `libsndfile` via system package manager is required. Example commands for popular distributions are listed below.
```bash
# ubuntu, debian
sudo apt-get install libsndfile1
# centos, fedora
sudo yum install libsndfile
# openSUSE
sudo zypper in libsndfile
```
For any problem with installtion of soundfile, please refer to [SoundFile](https://pypi.org/project/SoundFile/).
## Install Parakeet
There are two ways to install parakeet according to the purpose of using it.

 1. If you want to run experiments provided by parakeet or add new models and experiments, it is recommended to clone the project from github (Parakeet), and install it in editable mode.
       ```python
       git clone https://github.com/PaddlePaddle/Parakeet
       cd Parakeet
       pip install -e .
       ```
