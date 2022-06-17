# DeepSpeech2 to ONNX model

1. convert deepspeech2 model to ONNX, using Paddle2ONNX.
2. check paddleinference and onnxruntime output equal.
3. optimize onnx model
4. check paddleinference and optimized onnxruntime output equal.
5. quantize onnx model
4. check paddleinference and optimized onnxruntime output equal.

Please make sure [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) and [onnx-simplifier](https://github.com/zh794390558/onnx-simplifier/tree/dyn_time_shape) version is correct.

The example test with these packages installed:
```
paddle2onnx              0.9.8    # develop 62c5424e22cd93968dc831216fc9e0f0fce3d819
paddleaudio              0.2.1
paddlefsl                1.1.0
paddlenlp                2.2.6
paddlepaddle-gpu         2.2.2
paddlespeech             0.0.0       # develop
paddlespeech-ctcdecoders 0.2.0
paddlespeech-feat        0.1.0
onnx                     1.11.0
onnx-simplifier          0.0.0       # https://github.com/zh794390558/onnx-simplifier/tree/dyn_time_shape
onnxoptimizer            0.2.7
onnxruntime              1.11.0
```

## Using

```
bash run.sh --stage 0 --stop_stage 5
```

For more details please see `run.sh`.

## Outputs
The optimized onnx model is `exp/model.opt.onnx`, quanted model is `$exp/model.optset11.quant.onnx`.

To show the graph, please using `local/netron.sh`.


## [Results](https://github.com/PaddlePaddle/PaddleSpeech/wiki/ASR-Benchmark#streaming-asr)

机器硬件：`CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz`    
测试脚本：`Streaming Server`      

Acoustic Model | Model Size | enigne | dedoding_method | ctc_weight | decoding_chunk_size | num_decoding_left_chunk | RTF |
|:-------------:| :-----: | :-----: | :------------:| :-----: | :-----: | :-----: |:-----:|
| deepspeech2online_wenetspeech | 659MB | infernece | ctc_prefix_beam_search | - | 1 | - | 1.9108175171428279(utts=80) |
| deepspeech2online_wenetspeech | 659MB | onnx | ctc_prefix_beam_search | - | 1 | - | 0.5617182449999291 (utts=80) |
| deepspeech2online_wenetspeech | 166MB | onnx quant | ctc_prefix_beam_search | - | 1 | - | 0.44507715475808385 (utts=80) |

> quant 和机器有关，不是所有机器都支持。ONNX quant测试机器指令集支持:
> Flags:   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology eagerfpu pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat umip pku ospke avx512_vnni spec_ctrl