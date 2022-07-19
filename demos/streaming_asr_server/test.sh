# download the test wav
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav 

# read the wav and pass it to only streaming asr service
# If `127.0.0.1` is not accessible, you need to use the actual service IP address.
paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --input ./zh.wav

# read the wav and call streaming and punc service
# If `127.0.0.1` is not accessible, you need to use the actual service IP address.
paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --punc.server_ip 127.0.0.1 --punc.port 8190 --input ./zh.wav

