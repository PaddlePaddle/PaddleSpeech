train_output_path=$1
model_name=$2
weight_bits=$3

python3 ${BIN_DIR}/../PTQ_dynamic.py \
    --inference_dir ${train_output_path}/inference \
    --model_name ${model_name} \
    --weight_bits ${weight_bits}