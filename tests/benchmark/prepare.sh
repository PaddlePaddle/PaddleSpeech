source ../tools/venv/bin/activate

#Enter the example dir
pushd ../examples/aishell/s1

#Prepare the data
bash run.sh --stage 0 --stop_stage 0
