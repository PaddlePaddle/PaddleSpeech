#!/usr/bin/env bash
set -xe
# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){

    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    config_path=${2:-"conf/conformer.yaml"}
    output=${3:-"exp/conformer"}
    seed=${4:-"0"}
    ngpu=${5:-"1"}
    profiler_options=${6:-"None"}
    batch_size=${7:-"32"}
    fp_item=${8:-"fp32"}
    TRAIN_LOG_DIR=${9:-$(pwd)}

    benchmark_max_step=0

    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数

#   以下不用修改
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/recoder_${run_mode}_bs${batch_size}_${fp_item}_ngpu${ngpu}.txt
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"
    train_cmd="--config=${config_path}
               --output=${output}
               --seed=${seed}
               --nproc=${ngpu}
               --profiler-options "${profiler_options}"
               --benchmark-batch-size ${batch_size}
               --benchmark-max-step ${benchmark_max_step} "

    echo "run_mode "${run_mode}

    case ${run_mode} in
    sp) train_cmd="python3 -u ${BIN_DIR}/train.py "${train_cmd} ;;
    mp) train_cmd="python3 -u ${BIN_DIR}/train.py "${train_cmd} ;;
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
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

_set_params $@
_train
