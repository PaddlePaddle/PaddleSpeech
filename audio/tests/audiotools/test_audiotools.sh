python -m pip install -r ../../audiotools/requirements.txt
export PYTHONPATH=$PYTHONPATH:$(realpath ../../..) # this is root path of `PaddleSpeech`
wget  https://paddlespeech.bj.bcebos.com/PaddleAudio/audio_tools/audio.tar.gz
wget  https://paddlespeech.bj.bcebos.com/PaddleAudio/audio_tools/regression.tar.gz
tar -zxvf audio.tar.gz
tar -zxvf regression.tar.gz
python -m pytest