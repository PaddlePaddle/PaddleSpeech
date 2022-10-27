#!/bin/bash

if [ ! -f 000001-010000.txt ]; then
    wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/rhy_predict/000001-010000.txt
fi

if [ ! -f label_train-set.txt ]; then
    wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/rhy_predict/label_train-set.txt
fi


aishell_data=$1
csmsc_data=$2
processed_path=$3

python3 ./local/pre_for_sp_csmsc.py \
    --data=${csmsc_data} \
    --processed_path=${processed_path}

python3 ./local/pre_for_sp_aishell.py \
    --data=${aishell_data} \
    --processed_path=${processed_path}


echo "Finish data preparation."
exit 0
