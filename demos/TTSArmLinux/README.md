# PaddleSpeech TTS 文本到语音 ARM Linux Demo

修改自[demos/TTSAndroid](../TTSAndroid)，模型也来自该安卓Demo。

使用与安卓Demo版本相同的[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)推理库（[Paddle-Lite:68b66fd35](https://github.com/SwimmingTiger/Paddle-Lite/releases/tag/68b66fd35)），
该库兼容 Ubuntu 16.04 到 20.04，如果你的发行版与其不兼容，可以自行从源代码编译。

该Demo自带的模型与[Paddle-Lite 2.12](https://github.com/PaddlePaddle/Paddle-Lite/releases/tag/v2.12)不兼容，运行会崩溃，需要使用更新的版本。
不过如果换成用 Paddle-Lite 2.12 opt 工具优化的模型，应该可以兼容。

### 配置

打开 [config.sh](config.sh) 按需修改配置。

默认编译64位版本，如果要编译32位版本，把`ARM_ABI=armv8`改成`ARM_ABI=armv7hf`。

### 下载Paddle Lite库文件和模型文件

```
./download.sh
```

### 安装依赖

以 Ubuntu 18.04 为例：

```
sudo apt install build-essential cmake libopencv-dev
```

### 编译

```
./build.sh
```

### 运行

```
./run.sh
```

将把[src/main.cpp](src/main.cpp)里定义在`sentencesToChoose`数组中的十句话转换为`wav`文件，保存在`output`文件夹中。
