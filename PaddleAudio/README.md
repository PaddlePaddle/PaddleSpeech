# PaddleAudio:  The audio library for PaddlePaddle

## Introduction
PaddleAudio is the audio toolkit to speed up your audio research and development loop in PaddlePaddle. It currently provides a collection of audio datasets, feature-extraction functions, audio transforms,state-of-the-art pre-trained models in sound tagging/classification and anomaly sound detection. More models and features are on the roadmap.



## Features
- Spectrogram and related features are compatible with librosa.
- State-of-the-art models in sound tagging on Audioset, sound classification on esc50, and more to come.
- Ready-to-use audio embedding with a line of code, includes sound embedding and more on the roadmap.
- Data loading supports for common open source audio in multiple languages including English, Mandarin and so on.


## Install
```
git clone https://github.com/PaddlePaddle/models
cd models/PaddleAudio
pip install .

```

## Quick start
### Audio loading and feature extraction
```
import paddleaudio as pa
s,r = pa.load(f)
mel_spect = pa.melspectrogram(s,sr=r)
```

###  Examples
We provide a set of examples to help you get started in using PaddleAudio quickly.
- [PANNs:  acoustic scene and events analysis using pre-trained models](./examples/panns)
- [Environmental Sound classification on ESC-50 dataset](./examples/sound_classification)
- [Training a audio-tagging network on Audioset](./examples/audioset_training)

Please refer to [example directory](./examples) for more details.
