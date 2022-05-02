# download the test wav
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav 

# read the wav and pass it to only streaming asr service
python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --wavfile ./zh.wav

# read the wav and call streaming and punc service
python3 websocket_client.py --server_ip 127.0.0.1 --port 8290 --punc.server_ip 127.0.0.1 --punc.port 8190 --wavfile ./zh.wav
