#!/bin/bash
if [ $# != 1 ];then
    echo "usage: ${0} ckpt_dir"
    exit -1
fi

ckpt_dir=$1

stage=-1
stop_stage=100

source ${MAIN_ROOT}/utils/parse_options.sh

mkdir -p data
TARGET_DIR=${MAIN_ROOT}/examples/dataset
mkdir -p ${TARGET_DIR}

bash local/download_model.sh ${ckpt_dir}
if [ $? -ne 0 ]; then
   exit 1
fi

cd ${ckpt_dir}
tar xzvf aishell_model_v1.8_to_v2.x.tar.gz
cd -
mv ${ckpt_dir}/mean_std.npz data/
mv ${ckpt_dir}/vocab.txt data/


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # download data, generate manifests
    python3 ${TARGET_DIR}/aishell/aishell.py \
    --manifest_prefix="data/manifest" \
    --target_dir="${TARGET_DIR}/aishell"

    if [ $? -ne 0 ]; then
        echo "Prepare Aishell failed. Terminated."
        exit 1
    fi

    for dataset in train dev test; do
        mv data/manifest.${dataset} data/manifest.${dataset}.raw
    done
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # format manifest with tokenids, vocab size
    for dataset in train dev test; do
    {
        python3 ${MAIN_ROOT}/utils/format_data.py \
                --cmvn_path "data/mean_std.npz" \
                --unit_type "char" \
                --vocab_path="data/vocab.txt" \
                --manifest_path="data/manifest.${dataset}.raw" \
                --output_path="data/manifest.${dataset}"

        if [ $? -ne 0 ]; then
                echo "Formt mnaifest failed. Terminated."
                exit 1
        fi
    } &
    done
    wait
fi

echo "Aishell data preparation done."
exit 0
