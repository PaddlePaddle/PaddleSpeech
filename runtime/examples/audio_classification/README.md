# audio classification

This directory provieds audio classification on CPU

## conf
config is the input of engine

    [CONF]
    wav_normal=true
    wav_normal_type=linear
    wav_norm_mul_factor=1.0
    model_path=./inference.onnx
    param_path=
    dict_path=./label_list
    num_cpu_thread=1
    samp_freq=32000
    frame_length_ms=32
    frame_shift_ms=10
    num_bins=64
    low_freq=50
    high_freq=14000
    dither=0.0
## label_list
model output label

    Dog
    Rooster
    Pig
    Cow
    Frog
    Cat
    Hen
    Insects (flying)
    Sheep
    Crow
    Rain
    Sea waves
    Crackling fire
    .....
## scp && test.wav
scp is the input of engine and each line in scp is wav
## execute
../../build/Linux/x86_64/engine/audio_classification/nnet/panns_nnet_main --conf_path=./conf --scp_path=./scp --topk=1
usage: panns_nnet_main conf scp topk
output such as:

    wav_normal = true
    wav_normal_type = linear
    wav_norm_mul_factor = 1.0
    model_path = ./inference.onnx
    param_path = 
    dict_path = ./label_list
    num_cpu_thread = 1
    samp_freq = 32000
    frame_length_ms = 32
    frame_shift_ms = 10
    num_bins = 64
    low_freq = 50
    high_freq = 14000
    dither = 0.0
    [INFO] fastdeploy/runtime/runtime.cc(293)::CreateOrtBackend     Runtime initialized with Backend::ORT in Device::CPU.
    --- Init FastDeploy Runitme Done! 
    --- Model:  ./inference.onnx
    test.wav{"Clock alarm":"16.5309"}
## android demo
### install
#### copy lib & interface
cd ../../
sh build_android.sh
cp build/Android/arm64-v8a-api-21/cls-android-out/*.so examples/audio_classification/android_demo/app/src/main/cpp/jniLibs/arm64-v8a/
cp build/Android/arm64-v8a-api-21/cls-android-out/panns_interface.h examples/audio_classification/android_demo/app/src/main/cpp/
includes/

#### set path
push resource into android phone

1. change resource path in conf to gloabal path, such as:

    [CONF]
    wav_normal=true
    wav_normal_type=linear
    wav_norm_mul_factor=1.0
    model_path=/data/local/tmp/inference.onnx
    param_path=
    dict_path=/data/local/tmp/label_list
    num_cpu_thread=1
    samp_freq=32000
    frame_length_ms=32
    frame_shift_ms=10
    num_bins=64
    low_freq=50
    high_freq=14000
    dither=0.0
2. adb push conf label_list scp test.wav /data/local/tmp/
3. set reource path in android demo(android_demo/app/src/main/cpp/native-lib.cpp) to actual path, such as:

std::string conf_path = "/data/local/tmp/conf";
std::string wav_path = "/data/local/tmp/test.wav";

4. excecute android_demo in android studio
