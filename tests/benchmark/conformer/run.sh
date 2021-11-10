
# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
CUR_DIR=${PWD}
source ../../../tools/venv/bin/activate
#cd **
pushd ../../../examples/aishell/s1
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
# 2 拷贝该模型需要数据、预训练模型


source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

mkdir -p conf/benchmark
yq e ".training.accum_grad=1" conf/conformer.yaml > conf/benchmark/conformer.yaml

fp_item_list=(fp32)
bs_item=(16 30)
config_path=conf/benchmark/conformer.yaml
seed=0
output=exp/conformer
profiler_options=None
for fp_item in ${fp_item_list[@]}; do
    for batch_size in ${bs_item[@]}
        do
        rm exp -rf
        echo "index is speed, 8gpus, run_mode is multi_process, begin, conformer"
        run_mode=mp
        ngpu=8
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${config_path} ${output} ${seed} ${ngpu} ${profiler_options} ${batch_size} ${fp_item} ${CUR_DIR}
        rm exp -rf
        echo "index is speed, 1gpus, begin, conformer"
        run_mode=sp
        ngpu=1
        CUDA_VISIBLE_DEVICES=0 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${config_path} ${output} ${seed} ${ngpu} ${profiler_options} ${batch_size} ${fp_item} ${CUR_DIR}
    done
done

popd


