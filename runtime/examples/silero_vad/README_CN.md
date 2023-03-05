[English](README.md) | 简体中文
# Silero VAD 部署示例

本目录下提供`infer_onnx_silero_vad`快速完成 Silero VAD 模型在CPU/GPU。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上 VAD 推理为例，在本目录执行如下命令即可完成编译测试。

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载 VAD 模型文件和测试音频，解压后将模型和测试音频放置在与 infer_onnx_silero_vad.cc 同级目录下
wget https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad_sample.wav

# 推理
./infer_onnx_silero_vad ../silero_vad.onnx ../silero_vad_sample.wav
```

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:
- [如何在Windows中使用FastDeploy C++ SDK](../../../../docs/cn/faq/use_sdk_on_windows.md)

## VAD C++ 接口
### Vad 类

```c++
Vad::Vad(const std::string& model_file,
    const fastdeploy::RuntimeOption& custom_option = fastdeploy::RuntimeOption())
```

**参数**

> * **model_file**(str): 模型文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置

### setAudioCofig 函数

**必须在`init`函数前调用**

```c++
void Vad::setAudioCofig(int sr, int frame_ms, float threshold, int min_silence_duration_ms, int speech_pad_ms);
```

**参数**

> * **sr**(int): 采样率
> * **frame_ms**(int): 每次检测帧长，用于计算检测窗口大小
> * **threshold**(float): 结果概率判断阈值
> * **min_silence_duration_ms**(int): 用于计算判断是否是 silence 的阈值
> * **speech_pad_ms**(int): 用于计算 speach 结束时刻

### init 函数

用于初始化音频相关参数

```c++
void Vad::init();
```

### loadAudio 函数

加载音频

```c++
void Vad::loadAudio(const std::string& wavPath)
```

**参数**

> * **wavPath**(str): 音频文件路径

### Predict 函数

用于开始模型推理

```c++
bool Vad::Predict();
```

### getResult 函数

**用于获取推理结果**

```c++
std::vector<std::map<std::string, float>> Vad::getResult(
            float removeThreshold = 1.6, float expandHeadThreshold = 0.32, float expandTailThreshold = 0,
            float mergeThreshold = 0.3);
```

**参数**

> * **removeThreshold**(float): 丢弃结果片段阈值；部分识别结果太短则根据此阈值丢弃
> * **expandHeadThreshold**(float): 结果片段开始时刻偏移；识别到的开始时刻可能过于贴近发声部分，因此据此前移开始时刻
> * **expandTailThreshold**(float): 结果片段结束时刻偏移；识别到的结束时刻可能过于贴近发声部分，因此据此后移结束时刻
> * **mergeThreshold**(float): 有的结果片段十分靠近，可以合并成一个，据此合并发声片段

**输出结果格式为**`std::vector<std::map<std::string, float>>`

> 输出一个列表，每个元素是一个讲话片段
>
> 每个片段可以用 'start' 获取到开始时刻，用 'end' 获取到结束时刻

### 提示

1. `setAudioCofig`函数必须在`init`函数前调用
2. 输入的音频文件的采样率必须与代码中设置的保持一致

- [模型介绍](../)
- [如何切换模型推理后端引擎](../../../../docs/cn/faq/how_to_change_backend.md)
