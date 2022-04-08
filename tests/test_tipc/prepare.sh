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

if [ ${MODE} = "benchmark_train" ];then
    curPath=$(readlink -f "$(dirname "$0")")
        echo "curPath:"${curPath}
    cd ${curPath}/../..
    apt-get install libsndfile1 -y 
    pip install pytest-runner  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install kaldiio  -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install setuptools_scm -i https://pypi.tuna.tsinghua.edu.cn/simple 
    pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple 
    cd -
    if [ ${model_name} == "conformer" ]; then
        # set the URL for aishell_tiny dataset
        URL=${conformer_data_URL:-"None"}
        echo "URL:"${URL}
        if [ ${URL} == 'None' ];then
            echo "please contact author to get the URL.\n"
            exit
	else
	    wget -P ${curPath}/../../dataset/aishell/ ${URL} 
        fi
        sed -i "s#^URL_ROOT_TAG#URL_ROOT = '${URL}'#g" ${curPath}/conformer/scripts/aishell_tiny.py
        cp ${curPath}/conformer/scripts/aishell_tiny.py ${curPath}/../../dataset/aishell/
        cd ${curPath}/../../examples/aishell/asr1
        source path.sh
        # download audio data
        sed -i "s#aishell.py#aishell_tiny.py#g" ./local/data.sh
	sed -i "s#python3#python#g" ./local/data.sh
        bash ./local/data.sh || exit -1
        if [ $? -ne 0 ]; then
        exit 1
        fi
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

    if [ ${model_name} == "pwgan" ]; then
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

fi
