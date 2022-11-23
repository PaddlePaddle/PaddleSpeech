### Prepare the environment
Please follow the instructions shown in [here](../../../docs/source/install.md) to install the Deepspeech first.

### File list
└── benchmark                  # 模型名  
    ├── README.md              # 运行文档  
    ├── analysis.py        # log解析脚本,每个框架尽量统一,可参考[paddle的analysis.py](https://github.com/mmglove/benchmark/blob/jp_0907/scripts/analysis.py)  
    ├── recoder_mp_bs16_fp32_ngpu1.txt  # 单卡数据
    ├── recoder_mp_bs16_fp32_ngpu8.txt  # 8卡数据  
    ├── prepare.sh             #  竞品PyTorch运行环境搭建  
    ├── run_benchmark.sh       # 运行脚本（包含性能、收敛性）  
    ├── run_analysis_mp.sh     # 分析8卡的脚本  
    ├── run_analysis_sp.sh     # 分析单卡的脚本  
    ├── log
    │     ├── log_sp.out    # 单卡的结果
    │     └── log_mp.out    # 8卡的结果
    └── run.sh         # 全量运行脚本


### The physical environment
- 单机（单卡、8卡）
  - 系统：Ubuntu 16.04.6 LTS
  - GPU：Tesla V100-SXM2-16GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz * 96
  - Driver Version: 440.64.00
  - 内存：440 GB
  - CUDA、cudnn Version: cuda10.2-cudnn7
- 多机（32卡） TODO

### Docker 镜像,如:

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7`  
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

### Prepare the benchmark environment
```
bash prepare.sh
```

### Start benchmarking
```
bash run.sh
```

### The log
```
{"log_file": "recoder_sp_bs16_fp32_ngpu1.txt",
 "model_name": "Conformer",
 "mission_name": "one gpu",
 "direction_id": 1,
 "run_mode": "sp",
 "index": 1,
 "gpu_num": 1,
 "FINAL_RESULT": 23.228,
 "JOB_FAIL_FLAG": 0,
 "log_with_profiler": null,
 "profiler_path": null,
 "UNIT": "sent./sec"
}
```
