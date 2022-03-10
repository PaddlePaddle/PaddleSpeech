# 1. Prepare
First, install `line_profiler` via pip.
```sh
pip install line_profiler
```

# 2. Run
Run the specific script for profiling.
```sh
kernprof -l features/mel_spectrogram.py
python -m line_profiler -u 1e-3 mel_spectrogram.py.lprof
```

Result:
```sh
Timer unit: 0.001 s                                                                 

Total time: 22.1208 s
File: features/mel_spectrogram.py
Function: test_melspect_cpu at line 13
                                                                                                                                                                         
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    13                                           @profile
    14                                           def test_melspect_cpu(input_shape, times):
    15         1          0.1      0.1      0.0      paddle.set_device('cpu')
    16         1        234.5    234.5      1.1      x = paddle.randn(input_shape)
    17         1         85.3     85.3      0.4      feature_extractor = paddleaudio.features.MelSpectrogram(**feat_conf, dtype=x.dtype)
    18       101          0.5      0.0      0.0      for i in range(times):
    19       100      21800.5    218.0     98.6          y = feature_extractor(x)

Total time: 4.80543 s
File: features/mel_spectrogram.py
Function: test_melspect_gpu at line 22

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    22                                           @profile
    23                                           def test_melspect_gpu(input_shape, times):
    24         1          0.5      0.5      0.0      paddle.set_device('gpu')
    25         1       4144.8   4144.8     86.3      x = paddle.randn(input_shape)
    26         1         41.9     41.9      0.9      feature_extractor = paddleaudio.features.MelSpectrogram(**feat_conf, dtype=x.dtype)
    27       101          0.2      0.0      0.0      for i in range(times):
    28       100        618.1      6.2     12.9          y = feature_extractor(x)
```
