#!/bin/bash
# bash test.sh

StartService(){
    # Start service 
    paddlespeech_server start --config_file $config_file 1>>$log/server.log 2>>$log/server.log.wf &
    echo $! > pid

    start_num=$(cat $log/server.log.wf | grep "INFO:     Uvicorn running on http://" -c)
    flag="normal"
    while [[ $start_num -lt $target_start_num && $flag == "normal" ]]
    do
        start_num=$(cat $log/server.log.wf | grep "INFO:     Uvicorn running on http://" -c)
        # start service failed
        if [ $(cat $log/server.log.wf | grep -i "Failed to warm up on tts engine." -c) -gt $error_time ];then
            echo "Service started failed."  | tee -a $log/test_result.log
            error_time=$(cat $log/server.log.wf | grep -i "Failed to warm up on tts engine." -c)
            flag="unnormal"

        elif [ $(cat $log/server.log.wf | grep -i "AssertionError" -c) -gt $error_time ];then
            echo "Service started failed."  | tee -a $log/test_result.log
            error_time+=$(cat $log/server.log.wf | grep -i "AssertionError" -c)
            flag="unnormal"
        fi
    done
}

ClientTest_http(){
    for ((i=1; i<=3;i++))
    do
    python http_client.py --save_path ./out_http.wav 
    ((http_test_times+=1))
    done
}

ClientTest_ws(){
    for ((i=1; i<=3;i++))
    do
    python ws_client.py
    ((ws_test_times+=1))
    done
}

GetTestResult_http() {
    # Determine if the test was successful
    http_response_success_time=$(cat $log/server.log | grep "200 OK" -c)
    if (( $http_response_success_time == $http_test_times )) ; then
        echo "Testing successfully. $info"  | tee -a $log/test_result.log
    else
        echo "Testing failed. $info" | tee -a $log/test_result.log
    fi
    http_test_times=$http_response_success_time
}

GetTestResult_ws() {
    # Determine if the test was successful
    ws_response_success_time=$(cat $log/server.log.wf | grep "Complete the transmission of audio streams" -c)
    if (( $ws_response_success_time == $ws_test_times )) ; then
        echo "Testing successfully. $info"  | tee -a $log/test_result.log
    else
        echo "Testing failed. $info" | tee -a $log/test_result.log
    fi
    ws_test_times=$ws_response_success_time
}


engine_type=$1
log=$2
mkdir -p $log
rm -rf $log/server.log.wf 
rm -rf $log/server.log
rm -rf $log/test_result.log

config_file=./conf/application.yaml
server_ip=$(cat $config_file | grep "host" | awk -F " " '{print $2}')
port=$(cat $config_file | grep "port" | awk '/port:/ {print $2}')

echo "Sevice ip: $server_ip" | tee $log/test_result.log
echo "Sevice port: $port" | tee -a $log/test_result.log

# whether a process is listening on $port
pid=`lsof -i :"$port"|grep -v "PID" | awk '{print $2}'`
if [ "$pid" != "" ]; then
    echo "The port: $port is occupied, please change another port"
    exit
fi



target_start_num=0  # the number of start service
test_times=0  # The number of client test
error_time=0  # The number of error occurrences in the startup failure server.log.wf file

# start server: engine: tts_online, protocol: http, am: fastspeech2_cnndecoder_csmsc, voc: mb_melgan_csmsc
info="start server: engine: $engine_type, protocol: http, am: fastspeech2_cnndecoder_csmsc, voc: mb_melgan_csmsc."
echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_http
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_http
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log




python change_yaml.py --engine_type $engine_type --target_key voc --target_value hifigan_csmsc    # change voc: mb_melgan_csmsc -> hifigan_csmsc
# start server: engine: tts_online, protocol: http, am: fastspeech2_cnndecoder_csmsc, voc: hifigan_csmsc
info="start server: engine: $engine_type, protocol: http, am: fastspeech2_cnndecoder_csmsc, voc: hifigan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_http
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_http
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log



python change_yaml.py --engine_type $engine_type --target_key am --target_value fastspeech2_csmsc    # change am: fastspeech2_cnndecoder_csmsc -> fastspeech2_csmsc
# start server: engine: tts_online, protocol: http, am: fastspeech2_csmsc, voc: hifigan_csmsc
info="start server: engine: $engine_type, protocol: http, am: fastspeech2_csmsc, voc: hifigan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_http
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_http
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log


python change_yaml.py --engine_type $engine_type  --target_key voc --target_value mb_melgan_csmsc    # change voc: hifigan_csmsc -> mb_melgan_csmsc
# start server: engine: tts_online, protocol: http, am: fastspeech2_csmsc, voc: mb_melgan_csmsc
info="start server: engine: $engine_type, protocol: http, am: fastspeech2_csmsc, voc: mb_melgan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_http
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_http
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  
    
fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log


echo "********************************************* websocket **********************************************************"

python change_yaml.py --engine_type $engine_type --change_type protocol --target_key protocol --target_value websocket
# start server: engine: tts_online, protocol: websocket, am: fastspeech2_csmsc, voc: mb_melgan_csmsc
info="start server: engine: $engine_type, protocol: websocket, am: fastspeech2_csmsc, voc: mb_melgan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_ws
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_ws
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  
    
fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log

python change_yaml.py --engine_type $engine_type --target_key voc --target_value hifigan_csmsc    # change voc: mb_melgan_csmsc -> hifigan_csmsc
# start server: engine: tts_online, protocol: websocket, am: fastspeech2_csmsc, voc: hifigan_csmsc
info="start server: engine: $engine_type, protocol: websocket, am: fastspeech2_csmsc, voc: hifigan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_ws
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_ws
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log


python change_yaml.py --engine_type $engine_type --target_key am --target_value fastspeech2_cnndecoder_csmsc    # change am: fastspeech2_csmsc -> fastspeech2_cnndecoder_csmsc
# start server: engine: tts_online, protocol: websocket, am: fastspeech2_cnndecoder_csmsc, voc: hifigan_csmsc
info="start server: engine: $engine_type, protocol: websocket, am: fastspeech2_cnndecoder_csmsc, voc: hifigan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_ws
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_ws
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log



python change_yaml.py --engine_type $engine_type --target_key voc --target_value mb_melgan_csmsc    # change am: hifigan_csmsc -> mb_melgan_csmsc
# start server: engine: tts_online, protocol: websocket, am: fastspeech2_cnndecoder_csmsc, voc: mb_melgan_csmsc
info="start server: engine: $engine_type, protocol: websocket, am: fastspeech2_cnndecoder_csmsc, voc: mb_melgan_csmsc."

echo "$info"  | tee -a $log/test_result.log
((target_start_num+=1))
StartService

if [[ $start_num -eq $target_start_num && $flag == "normal" ]]; then
    echo "Service started successfully."  | tee -a $log/test_result.log
    ClientTest_ws
    echo "This round of testing is over."  | tee -a $log/test_result.log

    GetTestResult_ws
else
    echo "Service failed to start, no client test."
    target_start_num=$start_num  

fi

kill -9 `cat pid`
rm -rf pid
sleep 2s
echo "**************************************************************************************" | tee -a $log/test_result.log



echo "All tests completed."  | tee -a $log/test_result.log


# sohw all the test results
echo "***************** Here are all the test results ********************"
cat $log/test_result.log

# Restoring conf is the same as demos/speech_server
cp ./tts_online_application.yaml ./conf/application.yaml -rf
sleep 2s