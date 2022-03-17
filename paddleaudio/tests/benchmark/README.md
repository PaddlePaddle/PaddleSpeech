# 1. Prepare
First, install `pytest-benchmark` via pip.
```sh
pip install pytest-benchmark
```

# 2. Run
Run the specific script for profiling.
```sh
pytest melspectrogram.py
```

Result:
```sh
========================================================================== test session starts ==========================================================================
platform linux -- Python 3.7.7, pytest-7.0.1, pluggy-1.0.0
benchmark: 3.4.1 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /ssd3/chenxiaojie06/PaddleSpeech/DeepSpeech/paddleaudio
plugins: typeguard-2.12.1, benchmark-3.4.1, anyio-3.5.0
collected 4 items

melspectrogram.py ....                                                                                                                                            [100%]


-------------------------------------------------------------------------------------------------- benchmark: 4 tests -------------------------------------------------------------------------------------------------
Name (time in us)                        Min                    Max                   Mean              StdDev                 Median                 IQR            Outliers         OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_melspect_gpu_torchaudio        202.0765 (1.0)         360.6230 (1.0)         218.1168 (1.0)       16.3022 (1.0)         214.2871 (1.0)       21.8451 (1.0)          40;3  4,584.7001 (1.0)         286           1
test_melspect_gpu                   657.8509 (3.26)        908.0470 (2.52)        724.2545 (3.32)     106.5771 (6.54)        669.9096 (3.13)     113.4719 (5.19)          1;0  1,380.7300 (0.30)          5           1
test_melspect_cpu_torchaudio      1,247.6053 (6.17)      2,892.5799 (8.02)      1,443.2853 (6.62)     345.3732 (21.19)     1,262.7263 (5.89)     221.6385 (10.15)       56;53    692.8637 (0.15)        399           1
test_melspect_cpu                20,326.2549 (100.59)   20,607.8682 (57.15)    20,473.4125 (93.86)     63.8654 (3.92)     20,467.0429 (95.51)     68.4294 (3.13)          8;1     48.8438 (0.01)         29           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
========================================================================== 4 passed in 21.12s ===========================================================================

```
