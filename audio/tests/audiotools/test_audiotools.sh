python -m pip install -r ../audiotools/requirements.txt
wget -P ./test_data https://paddlespeech.bj.bcebos.com/PaddleAudio/audio_tools/audio.tar.gz
wget -P ./test_data https://paddlespeech.bj.bcebos.com/PaddleAudio/audio_tools/regression.tar.gz
find . -name "*✅.py" | xargs python -m pytest