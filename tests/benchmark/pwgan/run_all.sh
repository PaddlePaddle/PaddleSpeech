#!/usr/bin/env bash

stage=0
stop_stage=100

# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
cd ../../../
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
      sudo apt-get install libsndfile1
      pip install -e .
      pushd examples/csmsc/voc1
      source path.sh
      popd
fi
# 2 拷贝该模型需要数据、预训练模型
# 下载 baker 数据集到 home 目录下并解压缩到 home 目录下
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      wget https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar
      mkdir BZNSYP
      unrar x BZNSYP.rar BZNSYP
      wget https://paddlespeech.bj.bcebos.com/Parakeet/benchmark/durations.txt
fi
# 数据预处理
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

      python3 paddlespeech/t2s/exps/gan_vocoder/preprocess.py --rootdir=BZNSYP/ --dumpdir=dump --num-cpu=20 --cut-sil=True --dur-file=durations.txt --config=examples/csmsc/voc1/conf/default.yaml
      python3 utils/compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="feats"
      python3 paddlespeech/t2s/exps/gan_vocoder/normalize.py --metadata=dump/train/raw/metadata.jsonl --dumpdir=dump/train/norm --stats=dump/train/feats_stats.npy
      python3 paddlespeech/t2s/exps/gan_vocoder/normalize.py --metadata=dump/dev/raw/metadata.jsonl --dumpdir=dump/dev/norm --stats=dump/train/feats_stats.npy
      python3 paddlespeech/t2s/exps/gan_vocoder/normalize.py --metadata=dump/test/raw/metadata.jsonl --dumpdir=dump/test/norm --stats=dump/train/feats_stats.npy
fi
# 3 批量运行（如不方便批量，1，2需放到单个模型中）
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      model_mode_list=(pwg)
      fp_item_list=(fp32)
      # 满 bs 是 26
      bs_item_list=(6 26)
      for model_mode in ${model_mode_list[@]}; do
            for fp_item in ${fp_item_list[@]}; do
            for bs_item in ${bs_item_list[@]}; do
                  echo "index is speed, 1gpus, begin, ${model_name}"
                  run_mode=sp
                  CUDA_VISIBLE_DEVICES=0 bash tests/benchmark/pwgan/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 100 ${model_mode}     #  (5min)
                  sleep 60
                  echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
                  run_mode=mp
                  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 100 ${model_mode} 
                  sleep 60
                  done
            done
      done
fi