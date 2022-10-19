#!/bin/bash


aishell_data=$1
biaobei_data=$2
processed_path=$3

python3 ./local/pre_for_sp_biaobei.py \
    --data=${biaobei_data} \
    --processed_path=${processed_path}

python3 ./local/pre_for_sp_aishell.py \
    --data=${aishell_data} \
    --processed_path=${processed_path}


echo "Finish data preparation."
exit 0
