source ../tools/venv/bin/activate

#进入执行目录
pushd ../examples/aishell/s1

#准备数据
bash run.sh --stage 0 --stop_stage 0
