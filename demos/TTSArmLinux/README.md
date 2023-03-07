# TTS ARM Linux Demo

修改自 [demos/TTSAndroid](../TTSAndroid)，模型也来自该安卓 Demo。

### 配置编译选项

打开 [config.sh](config.sh) 按需修改配置。

默认编译 64 位版本，如果要编译 32 位版本，把 `ARM_ABI=armv8` 改成 `ARM_ABI=armv7hf` 。

### 安装依赖

```
# Ubuntu
sudo apt install build-essential cmake wget tar unzip

# CentOS
sudo yum groupinstall "Development Tools"
sudo yum install cmake wget tar unzip
```

### 下载 Paddle Lite 库文件和模型文件

预编译的二进制使用与安卓 Demo 版本相同的 Paddle Lite 推理库（[Paddle-Lite:68b66fd35](https://github.com/PaddlePaddle/Paddle-Lite/tree/68b66fd356c875c92167d311ad458e6093078449)）和模型（[fs2cnn_mbmelgan_cpu_v1.3.0](https://paddlespeech.bj.bcebos.com/demos/TTSAndroid/fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz)）。

可用以下命令下载：

```
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech/demos/TTSArmLinux
./download.sh
```

### 编译 Demo

```
./build.sh
```

预编译的二进制兼容 Ubuntu 16.04 到 20.04。

如果编译或链接失败，说明发行版与预编译库不兼容，请尝试手动编译 Paddle Lite 库，具体步骤在最下面。

### 运行

```
./run.sh
```

将把 [src/main.cpp](src/main.cpp) 里定义在 `sentencesToChoose` 数组中的十句话转换为 `wav` 文件，保存在 `output` 文件夹中。


## 手动编译 Paddle Lite 库

预编译的二进制兼容 Ubuntu 16.04 到 20.04，如果你的发行版与其不兼容，可以自行从源代码编译。

注意，我们只能保证 [Paddle-Lite:68b66fd35](https://github.com/PaddlePaddle/Paddle-Lite/tree/68b66fd356c875c92167d311ad458e6093078449) 与通过 `download.sh` 下载的模型兼容。
如果使用其他版本的 Paddle Lite 库，可能需要用对应版本的 opt 工具重新导出模型。

此外，[Paddle-Lite 2.12](https://github.com/PaddlePaddle/Paddle-Lite/releases/tag/v2.12) 与 TTS 不兼容，无法导出或运行 TTS 模型，需要使用更新的版本（比如 `develop` 分支中的代码）。
但 `develop` 分支中的代码可能与通过 `download.sh` 下载的模型不兼容，Demo 运行起来可能会崩溃。

### 安装 Paddle Lite 的编译依赖

```bash
# Ubuntu
sudo apt install build-essential cmake git python

# CentOS
sudo yum groupinstall "Development Tools"
sudo yum install cmake git python
```

### 编译 Paddle Lite 68b66fd35

```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout 68b66fd356c875c92167d311ad458e6093078449
./lite/tools/build_linux.sh --with_extra=ON
```

编译完成后，打开 Demo 的 [config.sh](config.sh)，把 `PADDLE_LITE_DIR` 改成以下值即可（注意替换 `/path/to/` 为实际目录）：

```
PADDLE_LITE_DIR="/path/to/Paddle-Lite/build.lite.linux.${ARM_ABI}.gcc/inference_lite_lib.armlinux.${ARM_ABI}/cxx"
```
