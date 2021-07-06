# [GigaSpeech](https://github.com/SpeechColab/GigaSpeech)

```
git clone https://github.com/SpeechColab/GigaSpeech.git

cd GigaSpeech
utils/gigaspeech_download.sh /disk1/audio_data/gigaspeech
toolkits/kaldi/gigaspeech_data_prep.sh --train-subset XL /disk1/audio_data/gigaspeech ../data
cd ..
```
