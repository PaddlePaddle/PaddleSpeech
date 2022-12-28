train_output_path=$1
model_name=$2

python3 ${BIN_DIR}/../../PTQ_static.py \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --inference_dir ${train_output_path}/inference \
    --model_name ${model_name} \
    --onnx_format=True 