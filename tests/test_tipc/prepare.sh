#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['benchmark_train_lite_infer' 'benchmark_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer', 'benchmark_train']


MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

echo "model_name:"${model_name}
trainer_list=$(func_parser_value "${lines[14]}")

if [[ ${MODE} = "benchmark_train" ]];then
    curPath=$(readlink -f "$(dirname "$0")")
    echo "curPath:"${curPath}    # /PaddleSpeech/tests/test_tipc
    cd ${curPath}/../..
    echo "------------- install for speech  "
    apt-get install libsndfile1 -y 
    pip install yacs -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install pytest-runner  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install kaldiio  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install setuptools_scm -i https://pypi.tuna.tsinghua.edu.cn/simple 
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple 
    pip install jsonlines
    pip list
    cd -
    if [[ ${model_name} == "conformer" ]]; then
        # set the URL for aishell_tiny dataset
        conformer_aishell_URL=${conformer_aishell_URL:-"None"}
        if [[ ${conformer_aishell_URL} == 'None' ]];then
            echo "please contact author to get the URL.\n"
            exit
	    else
            rm -rf ${curPath}/../../dataset/aishell/aishell.py
            rm -rf ${curPath}/../../dataset/aishell/data_aishell_tiny*
	        wget -P ${curPath}/../../dataset/aishell/ ${conformer_aishell_URL}
        fi
        cd ${curPath}/../../examples/aishell/asr1

        #Prepare the data
	    sed -i "s#python3#python#g" ./local/data.sh
        bash run.sh --stage 0 --stop_stage 0   # 执行第一遍的时候会偶现报错
        bash run.sh --stage 0 --stop_stage 0

        mkdir -p ${curPath}/conformer/benchmark_train/
        cp -rf conf ${curPath}/conformer/benchmark_train/
        cp -rf data ${curPath}/conformer/benchmark_train/
        cd ${curPath}

        sed -i "s#accum_grad: 2#accum_grad: 1#g" ${curPath}/conformer/benchmark_train/conf/conformer.yaml
        sed -i "s#data/#test_tipc/conformer/benchmark_train/data/#g" ${curPath}/conformer/benchmark_train/conf/conformer.yaml
        sed -i "s#conf/#test_tipc/conformer/benchmark_train/conf/#g" ${curPath}/conformer/benchmark_train/conf/conformer.yaml
        sed -i "s#data/#test_tipc/conformer/benchmark_train/data/#g" ${curPath}/conformer/benchmark_train/conf/tuning/decode.yaml
        sed -i "s#data/#test_tipc/conformer/benchmark_train/data/#g" ${curPath}/conformer/benchmark_train/conf/preprocess.yaml
    fi

    if [[ ${model_name} == "pwgan" ]]; then
        # 下载 csmsc 数据集并解压缩
        wget -nc https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar
        mkdir -p BZNSYP
        unrar x BZNSYP.rar BZNSYP
        wget -nc https://paddlespeech.bj.bcebos.com/Parakeet/benchmark/durations.txt
        # 数据预处理
        python ../paddlespeech/t2s/exps/gan_vocoder/preprocess.py --rootdir=BZNSYP/ --dumpdir=dump --num-cpu=20 --cut-sil=True --dur-file=durations.txt --config=../examples/csmsc/voc1/conf/default.yaml
        python ../utils/compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="feats"
        python ../paddlespeech/t2s/exps/gan_vocoder/normalize.py --metadata=dump/train/raw/metadata.jsonl --dumpdir=dump/train/norm --stats=dump/train/feats_stats.npy
        python ../paddlespeech/t2s/exps/gan_vocoder/normalize.py --metadata=dump/dev/raw/metadata.jsonl --dumpdir=dump/dev/norm --stats=dump/train/feats_stats.npy
        python ../paddlespeech/t2s/exps/gan_vocoder/normalize.py --metadata=dump/test/raw/metadata.jsonl --dumpdir=dump/test/norm --stats=dump/train/feats_stats.npy
    fi

    if [[ ${model_name} == "mdtc" ]]; then
        # 下载 Snips 数据集并解压缩
        wget https://paddlespeech.bj.bcebos.com/datasets/hey_snips_kws_4.0.tar.gz.1 
	wget https://paddlespeech.bj.bcebos.com/datasets/hey_snips_kws_4.0.tar.gz.2
        cat hey_snips_kws_4.0.tar.gz.* > hey_snips_kws_4.0.tar.gz
        rm hey_snips_kws_4.0.tar.gz.*
        tar -xzf hey_snips_kws_4.0.tar.gz
        # 解压后的数据目录 ./hey_snips_research_6k_en_train_eval_clean_ter
    fi

fi
