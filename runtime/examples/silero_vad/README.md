English | [简体中文](README_CN.md)

# Silero VAD Deployment Example

This directory provides examples that `infer_onnx_silero_vad` fast finishes the deployment of VAD models on CPU/GPU.

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

Taking VAD inference on Linux as an example, the compilation test can be completed by executing the following command in this directory.

```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the VAD model file and test audio. After decompression, place the model and test audio in the infer_onnx_silero_vad.cc peer directory
wget https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad_sample.wav

# inference
./infer_onnx_silero_vad ../silero_vad.onnx ../silero_vad_sample.wav
```

- The above command works for Linux or MacOS. Refer to:
  - [How to use FastDeploy C++ SDK in Windows](../../../../docs/en/faq/use_sdk_on_windows.md)  for SDK use-pattern in Windows

## VAD C++ Interface

### Vad Class

```c++
Vad::Vad(const std::string& model_file,
    const fastdeploy::RuntimeOption& custom_option = fastdeploy::RuntimeOption())
```

**Parameter**

> * **model_file**(str): Model file path
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default. (use the default configuration)

### setAudioCofig function

**Must be called before the `init` function**

```c++
void Vad::setAudioCofig(int sr, int frame_ms, float threshold, int min_silence_duration_ms, int speech_pad_ms);
```

**Parameter**

> * **sr**(int): sampling rate
> * **frame_ms**(int): The length of each detection frame, and it is used to calculate the detection window size
> * **threshold**(float): Result probability judgment threshold
> * **min_silence_duration_ms**(int): The threshold used to calculate whether it is silence
> * **speech_pad_ms**(int): Used to calculate the end time of the speech

### init function

Used to initialize audio-related parameters.

```c++
void Vad::init();
```

### loadAudio function

Load audio.

```c++
void Vad::loadAudio(const std::string& wavPath)
```

**Parameter**

> * **wavPath**(str): Audio file path

### Predict function

Used to start model reasoning.

```c++
bool Vad::Predict();
```

### getResult function

**Used to obtain reasoning results**

```c++
std::vector<std::map<std::string, float>> Vad::getResult(
            float removeThreshold = 1.6, float expandHeadThreshold = 0.32, float expandTailThreshold = 0,
            float mergeThreshold = 0.3);
```

**Parameter**

> * **removeThreshold**(float): Discard result fragment threshold; If some recognition results are too short, they will be discarded according to this threshold
> * **expandHeadThreshold**(float): Offset at the beginning of the segment; The recognized start time may be too close to the voice part, so move forward the start time accordingly
> * **expandTailThreshold**(float): Offset at the end of the segment; The recognized end time may be too close to the voice part, so the end time is moved back accordingly
> * **mergeThreshold**(float): Some result segments are very close and can be combined into one, and the vocal segments can be combined accordingly

**The output result format is**`std::vector<std::map<std::string, float>>`

> Output a list, each element is a speech fragment
>
> Each clip can use 'start' to get the start time and 'end' to get the end time

### Tips

1. `The setAudioCofig`function must be called before the `init` function
2. The sampling rate of the input audio file must be consistent with that set in the code

- [Model Description](../)
- [How to switch the model inference backend engine](../../../../docs/en/faq/how_to_change_backend.md)
