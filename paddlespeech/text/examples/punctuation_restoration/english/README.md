# 英文实验例程
## 测试数据：
- IWLST2012英文：test2011

## 运行代码
- 运行 `run.sh 0 0 conf/train_conf/bertBLSTM_base_en.yaml 1 conf/data_conf/english.yaml `


## 相关论文实验结果：
> * Nagy, Attila, Bence Bial, and Judit Ács. "Automatic punctuation restoration with BERT models." arXiv preprint arXiv:2101.07343 (2021)*  
>


## 实验结果：
- BertBLSTM
  - 实验配置：conf/train_conf/bertLinear_en.yaml
  - 测试结果：exp/bertLinear_enRe/checkpoints/3.pdparams

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  |0.667910   |0.715778   |0.822222   |0.735304  |
    |Recall     |0.755274   |0.868188   |0.804348   |0.809270  |
    |F1         |0.708911   |0.784651   |0.813187   |0.768916  |  
