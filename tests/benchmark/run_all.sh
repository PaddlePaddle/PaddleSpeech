#!/bin/bash

ROOT_DIR=../../

# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行：
# collect env info
bash ${ROOT_DIR}/utils/pd_env_collect.sh
cat pd_env.txt

# 执行目录：需说明
pushd ${ROOT_DIR}/examples/aishell/s1

# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
pushd ${ROOT_DIR}/tools; make; popd
source ${ROOT_DIR}/tools/venv/bin/activate
pushd ${ROOT_DIR}; bash setup.sh; popd


# 2 拷贝该模型需要数据、预训练模型
mkdir -p exp/log
loca/data.sh &> exp/log/data.log

# 3 批量运行（如不方便批量，1，2需放到单个模型中）

model_mode_list=(conformer)
fp_item_list=(fp32)
bs_item=(32 64 96)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}
            do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}     #  (5min)
            sleep 60
            echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            run_mode=mp
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
            sleep 60
            done
      done
done

popd # aishell/s1
