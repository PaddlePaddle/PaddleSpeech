python -m pip install -r ../../../paddlespeech/audiotools/requirements.txt
wget  https://paddlespeech.bj.bcebos.com/PaddleAudio/audio_tools/audio.tar.gz
wget  https://paddlespeech.bj.bcebos.com/PaddleAudio/audio_tools/regression.tar.gz
tar -zxvf audio.tar.gz
tar -zxvf regression.tar.gz
python -m pytest