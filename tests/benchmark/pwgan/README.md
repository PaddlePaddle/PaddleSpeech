执行
```bash
./run_all.sh
```
即可运行.
执行逻辑：
1. cd 到 ../../../ (也就是 Parakeet 目录)
2. 安装 parakeet 所需依赖
3. 从 bos 下载数据集并解压缩
4. 预处理数据集为训练 pwg 所需格式，保存到 Parakeet/dump 文件夹底下
5. 按照不同的参数执行 run_benchmark.sh 脚本
