dir=$1
exp_dir=$2
conf_path=$3

python3 ${BIN_DIR}/test.py \
        --config ${conf_path} \
        --data-dir ${dir} \
        --load-checkpoint ${exp_dir}