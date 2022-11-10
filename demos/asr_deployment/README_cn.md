([简体中文](./README_cn.md)|English)
# 基于SpeechX 的 ASR 部署 

## 简介

支持 U2/U2++/Deepspeech2 模型的 C++ 部署，其在工业实践中经常被用到。

更多 Speechx 信息可以参看[文档](../../speechx/README.md)。

## 使用
### 1. 环境

* python - 3.7
* docker - `registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7`
* os - Ubuntu 16.04.7 LTS
* gcc/g++/gfortran - 8.2.0
* cmake - 3.16.0

更多信息可以参看[文档](../../speechx/README.md)。

### 2. 编译 SpeechX

更多信息可以参看[文档](../../speechx/README.md)。

### 3. 例子
- Source path.sh
  ```bash
  source path.sh
  ```

- 下载模型并准备测试数据
  ```bash
  run.sh --stage 0 --stop_stage 0
  ```

- 解码
  
  ```bash
  # FP32
  ./loca/recognizer.sh

  # INT8
  ./local/recognizer_quant.sh
  ```

  输出:
  ```bash
  I1026 16:13:24.683531 48038 u2_recognizer_main.cc:55] utt: BAC009S0916W0495
  I1026 16:13:24.683578 48038 u2_recognizer_main.cc:56] wav dur: 4.17119 sec.
  I1026 16:13:24.683595 48038 u2_recognizer_main.cc:64] wav len (sample): 66739
  I1026 16:13:25.037652 48038 u2_recognizer_main.cc:87] Pratial result: 3 这令
  I1026 16:13:25.043697 48038 u2_recognizer_main.cc:87] Pratial result: 4 这令
  I1026 16:13:25.222124 48038 u2_recognizer_main.cc:87] Pratial result: 5 这令被贷款
  I1026 16:13:25.228385 48038 u2_recognizer_main.cc:87] Pratial result: 6 这令被贷款
  I1026 16:13:25.414669 48038 u2_recognizer_main.cc:87] Pratial result: 7 这令被贷款的员工
  I1026 16:13:25.420714 48038 u2_recognizer_main.cc:87] Pratial result: 8 这令被贷款的员工
  I1026 16:13:25.608129 48038 u2_recognizer_main.cc:87] Pratial result: 9 这令被贷款的员工们请
  I1026 16:13:25.801620 48038 u2_recognizer_main.cc:87] Pratial result: 10 这令被贷款的员工们请食难安
  I1026 16:13:25.804101 48038 feature_cache.h:44] set finished
  I1026 16:13:25.804128 48038 feature_cache.h:51] compute last feats done.
  I1026 16:13:25.948771 48038 u2_recognizer_main.cc:87] Pratial result: 11 这令被贷款的员工们请食难安
  I1026 16:13:26.246963 48038 u2_recognizer_main.cc:113] BAC009S0916W0495 这令被贷款的员工们请食难安
  ```

## 结果

> CER 测试集为 aishell-test
> RTF 计算包含提特征和解码
> 测试机器： Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz avx512_vnni

### FP32

```
Overall -> 5.75 % N=104765 C=99035 S=5587 D=143 I=294
Mandarin -> 5.75 % N=104762 C=99035 S=5584 D=143 I=294
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```

```
RTF is: 0.315337
```

### INT8

```
Overall -> 5.87 % N=104765 C=98909 S=5711 D=145 I=289
Mandarin -> 5.86 % N=104762 C=98909 S=5708 D=145 I=289
English -> 0.00 % N=0 C=0 S=0 D=0 I=0
Other -> 100.00 % N=3 C=0 S=3 D=0 I=0
```
