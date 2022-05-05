# download the test wav
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav 

# read the wav and pass it to service
python3 websocket_client.py --wavfile ./zh.wav
