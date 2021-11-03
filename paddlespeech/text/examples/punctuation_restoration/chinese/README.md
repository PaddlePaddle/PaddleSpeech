# 中文实验例程
## 测试数据：
- IWLST2012中文：test2012

## 运行代码
- 运行 `run.sh 0 0 conf/train_conf/bertBLSTM_zh.yaml 1 conf/data_conf/chinese.yaml `

## 实验结果：
- BertLinear
  - 实验配置：conf/train_conf/bertLinear_zh.yaml
  - 测试结果

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  | 0.425665  | 0.335190  | 0.698113  | 0.486323 |  
    |Recall     | 0.511278  | 0.572108  | 0.787234  | 0.623540 |  
    |F1         | 0.464560  | 0.422717  | 0.740000  | 0.542426 |  

- BertBLSTM
  - 实验配置：conf/train_conf/bertBLSTM_zh.yaml
  - 测试结果 avg_1

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  |  0.469484 | 0.550604  | 0.801887  | 0.607325 |
    |Recall     |  0.580271 | 0.592408  | 0.817308  | 0.663329 |
    |F1         |  0.519031 | 0.570741  | 0.809524  | 0.633099 |  

  - BertBLSTM/avg_1测试标贝合成数据

    |           | COMMA     | PERIOD    | QUESTION  | OVERALL  |  
    |-----------|-----------|-----------|-----------|--------- |  
    |Precision  |  0.217192 | 0.196339  | 0.820717  | 0.411416 |
    |Recall     |  0.205922 | 0.892531  | 0.416162  | 0.504872 |
    |F1         |  0.211407 | 0.321873  | 0.552279  | 0.361853 |
