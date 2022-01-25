#! /usr/bin/env bash

model=$1
data=$2
config_conf=$3
embedding=$4

echo "model: ${model}"
echo "data: ${data}"

python3 ./local/extract_vector.py \
                    --model ${model} \
                    --data ${data} \
                    --config ${config_conf} \
                    --spker-embedding ${embedding}