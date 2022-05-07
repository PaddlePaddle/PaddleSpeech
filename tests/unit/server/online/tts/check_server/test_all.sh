#!/bin/bash
# bash test_all.sh

log_all_dir=./log

cp ./tts_online_application.yaml ./conf/application.yaml -rf

bash test.sh tts_online $log_all_dir/log_tts_online_cpu

python change_yaml.py --change_type engine_type --target_key engine_list --target_value tts_online-onnx
bash test.sh tts_online-onnx $log_all_dir/log_tts_online-onnx_cpu


python change_yaml.py --change_type device --target_key device --target_value gpu:3
bash test.sh tts_online $log_all_dir/log_tts_online_gpu

python change_yaml.py --change_type engine_type --target_key engine_list --target_value tts_online-onnx
python change_yaml.py --change_type device --target_key device --target_value gpu:3
bash test.sh tts_online-onnx $log_all_dir/log_tts_online-onnx_gpu 

echo "************************************** show all test results ****************************************"
cat $log_all_dir/log_tts_online_cpu/test_result.log
cat $log_all_dir/log_tts_online-onnx_cpu/test_result.log
cat $log_all_dir/log_tts_online_gpu/test_result.log
cat $log_all_dir/log_tts_online-onnx_gpu/test_result.log
