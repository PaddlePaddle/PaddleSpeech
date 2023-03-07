# Silero VAD - pre-trained enterprise-grade Voice Activity Detector

This directory provides VAD models on CPU/GPU.

![](https://user-images.githubusercontent.com/36505480/198026365-8da383e0-5398-4a12-b7f8-22c2c0059512.png)

## Linux

### Build Runtime
```bash
# cd /path/to/paddlespeech/runtime
cmake -B build -DBUILD_SHARED_LIBS=OFF -DWITH_ASR=OFF -DWITH_CLS=OFF -DWITH_VAD=ON
cmake --build build
```

Since VAD using FastDeploy runtime, if you have another FastDeploy Library, you can using this command to build:

```bash
# cd /path/to/paddlespeech/runtime
cmake -B build -DBUILD_SHARED_LIBS=OFF -DWITH_ASR=OFF -DWITH_CLS=OFF -DWITH_VAD=ON -DFASTDEPLOY_INSTALL_DIR=/workspace/zhanghui/paddle/FastDeploy/build/Linux/x86_64/install
cmake --build build
```

`DFASTDEPLOY_INSTALL_DIR` is the directory of FastDeploy Library.

### Run Demo

After building success, we can do this to run demo under this example dir:

```bash 
bash run.sh
```

### Result

```bash
/workspace/zhanghui/PaddleSpeech/runtime/engine/vad/nnet/vad.cc(92)::SetConfig  sr=16000 threshold=0.45 frame_ms=32 min_silence_duration_ms=200 speech_pad_left_ms=0 speech_pad_right_ms=0[INFO] fastdeploy/runtime/runtime.cc(293)::CreateOrtBackend     Runtime initialized with Backend::ORT in Device::CPU./workspace/zhanghui/PaddleSpeech/runtime/engine/vad/nnet/vad.cc(141)::Initialize        init done.
[SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [STA] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [END] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [STA] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SIL] [SIL] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [END] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [STA] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [END] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [STA] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SIL] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SPE] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [END] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] [SIL] speak start: 0.32 s, end: 2.496 s | speak start: 3.296 s, end: 4.672 s | speak start: 5.408 s, end: 7.936 s | speak start: 8.192 s, end: 10.72 s   
vad_nnet_main done!

sr = 16000
frame_ms = 32
threshold = 0.45
min_silence_duration_ms = 200
speech_pad_left_ms = 200
speech_pad_right_ms = 0
model_path = ./data/silero_vad/silero_vad.onnx
param_path = (default)
num_cpu_thread = 1(default)
/workspace/zhanghui/PaddleSpeech/runtime/engine/vad/nnet/vad.cc(92)::SetConfig  sr=16000 threshold=0.45 frame_ms=32 min_silence_duration_ms=200 speech_pad_left_ms=200 speech_pad_right_ms=0
[INFO] fastdeploy/runtime/runtime.cc(293)::CreateOrtBackend     Runtime initialized with Backend::ORT in Device::CPU.
/workspace/zhanghui/PaddleSpeech/runtime/engine/vad/nnet/vad.cc(141)::Initialize        init done.
1 1 1 1 1 1 1 1 1 1 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 4 1 1 1 1 1 1 1 1 2 3 3 3 3 3 3 3 3 3 3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 4 1 1 1 1 1 1 1 1 1 vad_interface_main done!
```

The environment as below:

```text
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              80
On-line CPU(s) list: 0-79
Thread(s) per core:  2
Core(s) per socket:  20
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6Model:               85Model name:          Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
Stepping:            7
CPU MHz:             2599.998
BogoMIPS:            5199.99
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32KL1i cache:           32KL2 cache:            1024K
L3 cache:            33792K
NUMA node0 CPU(s):   0-39
NUMA node1 CPU(s):   40-79
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology nonstop_tsc eagerfpu pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat umip pku ospke avx512_vnni spec_ctrl arch_capabilities
```

## Android

When to using on Android, please setup your `NDK` enverment before, then do as below:

```bash
# cd /path/to/paddlespeech/runtime
bash build_android.sh
```

## VAD Interface

For vad interface please see [](../../engine/vad/interface/).

### Create Handdle

```c++
PPSHandle_t PPSVadCreateInstance(const char* conf_path);
```

### Destroy Handdle

```c++
int PPSVadDestroyInstance(PPSHandle_t instance);
```

### Reset Vad State

```c++
int PPSVadReset(PPSHandle_t instance);
```

Reset Vad state before processing next `wav`.

### Get Chunk Size

```c++
int PPSVadChunkSizeSamples(PPSHandle_t instance);
```

This API will return chunk size in `sample` unit.
When do forward, we need feed `chunk size` samples, except last chunk.

### Vad Forward

```c++
PPSVadState_t PPSVadFeedForward(PPSHandle_t instance,
                                float* chunk,
                                int num_element);
```

Vad has below states:
```c++
typedef enum {
    PPS_VAD_ILLEGAL = 0,  // error
    PPS_VAD_SIL,          // silence
    PPS_VAD_START,        // start speech
    PPS_VAD_SPEECH,       // in speech
    PPS_VAD_END,          // end speech
    PPS_VAD_NUMSTATES,    // number of states
} PPSVadState_t;
```

If `PPSVadFeedForward` occur an error will return `PPS_VAD_ILLEGAL` state.


## FastDeploy Runtime

For FastDeploy software and hardware requements, and pre-released library please to see [FastDeploy](https://github.com/PaddlePaddle/FastDeploy):

- 1. [FastDeploy Environment Requirements](https://github.com/PaddlePaddle/FastDeploy/docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. [FastDeploy Precompiled Library](https://github.com/PaddlePaddle/FastDeploy/docs/en/build_and_install/download_prebuilt_libraries.md).


## Download Pre-trained ONNX Model

For developers' testing, model exported by VAD are provided below. Developers can download them directly.

| 模型                                                         | 大小  | 备注                                                         |
| :----------------------------------------------------------- | :---- | :----------------------------------------------------------- |
| [silero-vad](https://bj.bcebos.com/paddlehub/fastdeploy/silero_vad.tgz) | 1.8MB | This model file is sourced from [snakers4/silero-vad](https://github.com/snakers4/silero-vad)，MIT License |


## Reference
* https://github.com/snakers4/silero-vad
* https://github.com/PaddlePaddle/FastDeploy/blob/develop/examples/audio/silero-vad/README.md
