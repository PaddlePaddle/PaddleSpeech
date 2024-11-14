#!/bin/bash
# bash test_server_client.sh
## require lsof to get server pid
## apt-get install -y lsof

StartService(){
    # Start service 
    paddlespeech_server start --config_file $config_file 1>>log/server.log 2>>log/server.log.wf &
    echo $! > pid

    start_num=$(cat log/server.log.wf | grep "INFO:     Uvicorn running on http://" -c)
    flag="normal"
    while [[ $start_num -lt $target_start_num && $flag == "normal" ]]
    do
        start_num=$(cat log/server.log.wf | grep "INFO:     Uvicorn running on http://" -c)
        # start service failed
        if [ $(cat log/server.log.wf | grep -i "error" -c) -gt $error_time ];then
            echo "Service started failed."  | tee -a ./log/test_result.log
            error_time=$(cat log/server.log.wf | grep -i "error" -c)
            flag="unnormal"
        fi
    done
}

ClientTest(){
    # Client test
    # test asr client
    paddlespeech_client asr --server_ip $server_ip --port $port --input ./zh.wav 
    ((test_times+=1))
    paddlespeech_client asr --server_ip $server_ip --port $port --input ./zh.wav 
    ((test_times+=1))

    # test tts client
    paddlespeech_client tts --server_ip $server_ip --port $port --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav 
    ((test_times+=1))
    paddlespeech_client tts --server_ip $server_ip --port $port --input "您好，欢迎使用百度飞桨语音合成服务。" --output output.wav 
    ((test_times+=1))  

    # test cls client
    paddlespeech_client cls --server_ip $server_ip --port $port --input ./zh.wav 
    ((test_times+=1))
    paddlespeech_client cls --server_ip $server_ip --port $port --input ./zh.wav 
    ((test_times+=1)) 

}

GetTestResult() {
    # Determine if the test was successful
    response_success_time=$(cat log/server.log | grep "200 OK" -c)
    if (( $response_success_time == $test_times )) ; then
        echo "Testing successfully. The service configuration is: asr engine type: $1; tts engine type: $1; cls engine type: $1; device: $2."  | tee -a ./log/test_result.log
    else
        echo "Testing failed. The service configuration is: asr engine type: $1; tts engine type: $1; cls engine type: $1; device: $2." | tee -a ./log/test_result.log
    fi
    test_times=$response_success_time
}


mkdir -p log
rm -rf log/server.log.wf 
rm -rf log/server.log
rm -rf log/test_result.log

cp ../../../../demos/speech_server/conf/application.yaml ./conf/
config_file=./conf/application.yaml
server_ip=$(cat $config_file | grep "host" | awk -F " " '{print $2}')
port=$(cat $config_file | grep "port" | awk '/port:/ {print $2}')

echo "Sevice ip: $server_ip" | tee ./log/test_result.log
echo "Sevice port: $port" | tee -a ./log/test_result.log

# whether a process is listening on $port
pid=`lsof -i :"$port"|grep -v "PID" | awk '{print $2}'`
if [ "$pid" != "" ]; then
    echo "The port: $port is occupied, please change another port"
    exit
fi

# download test audios for ASR client
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav


target_start_num=0  # the number of start service
test_times=0  # The number of client test
error_time=0  # The number of error occurrences in the startup failure server.log.wf file

# start server: asr engine type: python; tts engine type: python; cls engine type: python; device: gpu
echo "Start the service: asr engine type: python; tts engine type: python; cls engine type: python; device: gpu"  | tee -a ./log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a ./log/test_result.log
    ClientTest
    echo "This round of testing is over."  | tee -a ./log/test_result.log

    GetTestResult python gpu
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a ./log/test_result.log



# start server: asr engine type: python; tts engine type: python; cls engine type: python; device: cpu
python change_yaml.py --change_task device-asr_python-cpu    # change asr_python device: cpu
python change_yaml.py --change_task device-tts_python-cpu    # change tts_python device: cpu
python change_yaml.py --change_task device-cls_python-cpu    # change cls_python device: cpu

echo "Start the service: asr engine type: python; tts engine type: python; cls engine type: python; device: cpu"  | tee -a ./log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a ./log/test_result.log
    ClientTest
    echo "This round of testing is over."  | tee -a ./log/test_result.log

    GetTestResult python cpu
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a ./log/test_result.log


# start server: asr engine type: inference; tts engine type: inference; cls engine type: inference; device: gpu
python change_yaml.py --change_task enginetype-asr_inference    # change engine_list: 'asr_python' -> 'asr_inference'
python change_yaml.py --change_task enginetype-tts_inference    # change engine_list: 'tts_python' -> 'tts_inference'
python change_yaml.py --change_task enginetype-cls_inference    # change engine_list: 'cls_python' -> 'cls_inference'

echo "Start the service: asr engine type: inference; tts engine type: inference; cls engine type: inference; device: gpu"  | tee -a ./log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a ./log/test_result.log
    ClientTest
    echo "This round of testing is over."  | tee -a ./log/test_result.log

    GetTestResult inference gpu
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a ./log/test_result.log


# start server: asr engine type: inference; tts engine type: inference; cls engine type: inference; device: cpu
python change_yaml.py --change_task device-asr_inference-cpu    # change asr_inference device: cpu
python change_yaml.py --change_task device-tts_inference-cpu    # change tts_inference device: cpu
python change_yaml.py --change_task device-cls_inference-cpu    # change cls_inference device: cpu

echo "start the service: asr engine type: inference; tts engine type: inference; cls engine type: inference; device: cpu"  | tee -a ./log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a ./log/test_result.log
    ClientTest
    echo "This round of testing is over."  | tee -a ./log/test_result.log

    GetTestResult inference cpu
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  
    
fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a ./log/test_result.log

echo "All tests completed."  | tee -a ./log/test_result.log

# sohw all the test results
echo "***************** Here are all the test results ********************"
cat ./log/test_result.log

# Restoring conf is the same as demos/speech_server
cp ../../../../demos/speech_server/conf/application.yaml ./conf/
