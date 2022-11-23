cd ../../../
pip install -e .   # 安装pdspeech
cd -
#Enter the example dir
pushd ../../../examples/aishell/asr1

#Prepare the data
bash run.sh --stage 0 --stop_stage 0
