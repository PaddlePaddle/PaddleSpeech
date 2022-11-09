train_output_path=$1
model_dir=$2
output_dir=$3
model=$4
valid_targets=$5

model_name=${model%_*}
echo model_name: ${model_name}



mkdir -p ${train_output_path}/${output_dir}

paddle_lite_opt \
    --model_file ${train_output_path}/${model_dir}/${model}.pdmodel \
    --param_file  ${train_output_path}/${model_dir}/${model}.pdiparams \
    --optimize_out ${train_output_path}/${output_dir}/${model}_${valid_targets} \
    --valid_targets ${valid_targets}
