# DeepSpeech2 on PaddlePaddle

*DeepSpeech2 on PaddlePaddle* is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine, based on [Baidu's Deep Speech 2 paper](http://proceedings.mlr.press/v48/amodei16.pdf), with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform. Our vision is to empower both industrial application and academic research on speech-to-text, via an easy-to-use, efficent and scalable integreted implementation, including training & inferencing module, distributed [PaddleCloud](https://github.com/PaddlePaddle/cloud) training, and demo deployment. Besides, several pre-trained models for both English and Mandarin speech are also released.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Training a Model](#training-a-model)
- [Inference and Evaluation](#inference-and-evaluation)
- [Distributed Cloud Training](#distributed-cloud-training)
- [Hyper-parameters Tuning](#hyper-parameters-tuning)
- [Trying Live Demo with Your Own Voice](#trying-live-demo-with-your-own-voice)
- [Experiments and Benchmarks](#experiments-and-benchmarks)
- [Questions and Help](#questions-and-help)

## Prerequisites
- Only support Python 2.7
- PaddlePaddle the latest version (please refer to the [Installation Guide](https://github.com/PaddlePaddle/Paddle#installation))

## Installation

Please install the [prerequisites](#prerequisites) above before moving on this.

```
git clone https://github.com/PaddlePaddle/models.git
cd models/deep_speech_2
sh setup.sh
```

## Getting Started

TODO

## Data Preparation

```
cd datasets
sh run_all.sh
cd ..
```

`sh run_all.sh` prepares all ASR datasets (currently, only LibriSpeech available). After running, we have several summarization manifest files in json-format.

A manifest file summarizes a speech data set, with each line containing the meta data (i.e. audio filepath, transcript text, audio duration) of each audio file within the data set, in json format. Manifest file serves as an interface informing our system of  where and what to read the speech samples.


More help for arguments:

```
python datasets/librispeech/librispeech.py --help
```



```
python tools/compute_mean_std.py
```

It will compute mean and stdandard deviation for audio features, and save them to a file with a default name `./mean_std.npz`. This file will be used in both training and inferencing. The default feature of audio data is power spectrum, and the mfcc feature is also supported. To train and infer based on mfcc feature, please generate this file by

```
python tools/compute_mean_std.py --specgram_type mfcc
```

and specify ```--specgram_type mfcc``` when running train.py, infer.py, evaluator.py or tune.py.

More help for arguments:

```
python tools/compute_mean_std.py --help
```

## Training a model

For GPU Training:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py
```

For CPU Training:

```
python train.py --use_gpu False
```

More help for arguments:

```
python train.py --help
```

### Inference and Evaluation

The following steps, inference, parameters tuning and evaluating, will require a language model during decoding.
A compressed language model is provided and can be accessed by

```
cd ./lm
sh run.sh
cd ..
```



For GPU inference

```
CUDA_VISIBLE_DEVICES=0 python infer.py
```

For CPU inference

```
python infer.py --use_gpu=False
```

More help for arguments:

```
python infer.py --help
```


```
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```

More help for arguments:

```
python evaluate.py --help
```

## Hyper-parameters Tuning

Usually, the parameters $\alpha$ and $\beta$ for the CTC [prefix beam search](https://arxiv.org/abs/1408.2873) decoder need to be tuned after retraining the acoustic model.

For GPU tuning

```
CUDA_VISIBLE_DEVICES=0 python tune.py
```

For CPU tuning

```
python tune.py --use_gpu=False
```

More help for arguments:

```
python tune.py --help
```

Then reset parameters with the tuning result before inference or evaluating.

## Distributed Cloud Training

If you wish to train DeepSpeech2 on PaddleCloud, please refer to
[Train DeepSpeech2 on PaddleCloud](https://github.com/PaddlePaddle/models/tree/develop/deep_speech_2/cloud).

## Trying Live Demo with Your Own Voice

A real-time ASR demo is built for users to try out the ASR model with their own voice. Please do the following installation on the machine you'd like to run the demo's client (no need for the machine running the demo's server).

For example, on MAC OS X:

```
brew install portaudio
pip install pyaudio
pip install pynput
```
After a model and language model is prepared, we can first start the demo's server:

```
CUDA_VISIBLE_DEVICES=0 python demo_server.py
```
And then in another console, start the demo's client:

```
python demo_client.py
```
On the client console, press and hold the "white-space" key on the keyboard to start talking, until you finish your speech and then release the "white-space" key. The decoding results (infered transcription) will be displayed.

It could be possible to start the server and the client in two seperate machines, e.g. `demo_client.py` is usually started in a machine with a microphone hardware, while `demo_server.py` is usually started in a remote server with powerful GPUs. Please first make sure that these two machines have network access to each other, and then use `--host_ip` and `--host_port` to indicate the server machine's actual IP address (instead of the `localhost` as default) and TCP port, in both `demo_server.py` and `demo_client.py`.

## Experiments and Benchmarks

## Questions and Help
