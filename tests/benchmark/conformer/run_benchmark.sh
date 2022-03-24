#!/usr/bin/env bash
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){

    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    config_path=${2:-"conf/conformer.yaml"}
    decode_config_path=${3:-"conf/tuning/decode.yaml"}
    output=${4:-"exp/conformer"}
    seed=${5:-"0"}
    ngpu=${6:-"1"}
    profiler_options=${7:-"None"}
    batch_size=${8:-"32"}
    fp_item=${9:-"fp32"}
    model_item=${10:-"conformer"}
    benchmark_max_step=0
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数
# 添加日志解析需要的参数
    base_batch_size=${batch_size}
    mission_name="语音识别"
    direction_id="1"
    ips_unit="sent./sec"
    skip_steps=10                     # 解析日志，有些模型前几个step耗时长，需要跳过                                    (必填)
    keyword="ips:"                 # 解析日志，筛选出数据所在行的关键字                                             (必填)
    index="1"
    model_name=${model_item}_bs${batch_size}_${fp_item}

#   以下不用修改
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/recoder_${model_item}_${run_mode}_bs${batch_size}_${fp_item}_ngpu${ngpu}
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    train_cmd="--config=${config_path} \
           --decode_cfg=${decode_config_path} \
           --output=${output} \
           --seed=${seed} \
           --ngpu=${ngpu} \
           --benchmark-batch-size ${batch_size} \
           --benchmark-max-step ${benchmark_max_step} "
    if [ ${profiler_options} != "None" ]; then
        train_cmd=${train_cmd}" --profiler-options=${profiler_options}"
    fi

    case ${run_mode} in
    sp) train_cmd="python -u ${BIN_DIR}/train.py "${train_cmd} ;;
    mp) rm -rf ./mylog
        train_cmd="python -u ${BIN_DIR}/train.py "${train_cmd} ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac
    echo ${train_cmd}
# 以下不用修改
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;该脚本在连调时可从benchmark repo中下载https://github.com/PaddlePaddle/benchmark/blob/master/scripts/run_model.sh;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只想产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只想要产出训练log可以注掉本行,提交时需打开

