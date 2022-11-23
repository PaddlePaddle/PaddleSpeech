
# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
CUR_DIR=${PWD} # PaddleSpeech/tests/benchmark/conformer
cd ../../../
log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}  #  benchmark系统指定该参数,不需要跑profile时,log_path指向存speed的目录
cd ${CUR_DIR}
sed -i '/set\ -xe/d' run_benchmark.sh

#cd **
pushd ../../../examples/aishell/asr1
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
# 2 拷贝该模型需要数据、预训练模型


source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;
mkdir -p conf/benchmark
#yq e ".training.accum_grad=1" conf/conformer.yaml > conf/benchmark/conformer.yaml
cp conf/conformer.yaml  conf/benchmark/conformer.yaml
sed -i "s/  accum_grad: 2/  accum_grad: 1/g" conf/benchmark/conformer.yaml
fp_item_list=(fp32)
bs_item=(16)
config_path=conf/benchmark/conformer.yaml
decode_config_path=conf/tuning/decode.yaml
seed=0
output=exp/conformer
profiler_options=None
model_item=conformer
for fp_item in ${fp_item_list[@]}; do
    for bs_item in ${bs_item[@]}
        do
        rm exp -rf
        log_name=speech_${model_item}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
        echo "index is speed, 8gpus, run_mode is multi_process, begin, conformer"
        run_mode=mp
        ngpu=8
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${config_path} ${decode_config_path} ${output} ${seed} ${ngpu} ${profiler_options} ${bs_item} ${fp_item} ${model_item} | tee ${log_path}/${log_name}_speed_8gpus8p 2>&1
        sleep 60
        log_name=speech_${model_item}_bs${bs_item}_${fp_item}   # 如:clas_MobileNetv1_mp_bs32_fp32_8
        echo "index is speed, 1gpus, begin, ${log_name}"
        run_mode=sp
        ngpu=1
        CUDA_VISIBLE_DEVICES=0 bash ${CUR_DIR}/run_benchmark.sh ${run_mode} ${config_path} ${decode_config_path} ${output} ${seed} ${ngpu} ${profiler_options} ${bs_item} ${fp_item} ${model_item} | tee ${log_path}/${log_name}_speed_1gpus 2>&1   #  (5min)
        sleep 60
    done
done

popd


