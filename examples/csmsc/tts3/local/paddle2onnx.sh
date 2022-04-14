train_output_path=$1
model_dir=$2
output_dir=$3
model=$4

enable_dev_version=True

model_name=${model%_*}
echo model_name: ${model_name}

if [ ${model_name} = 'mb_melgan' ] ;then
    enable_dev_version=False
fi

mkdir -p ${train_output_path}/${output_dir}

paddle2onnx \
    --model_dir ${train_output_path}/${model_dir} \
    --model_filename ${model}.pdmodel \
    --params_filename ${model}.pdiparams \
    --save_file ${train_output_path}/${output_dir}/${model}.onnx \
    --opset_version 11 \
    --enable_dev_version ${enable_dev_version}