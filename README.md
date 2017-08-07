# Deep Speech 2 on PaddlePaddle

## Installation

### Prerequisites

 - **Python = 2.7** only supported;
 - **cuDNN >= 6.0** is required to utilize NVIDIA GPU platform in the installation of PaddlePaddle, and the **CUDA toolkit** with proper version suitable for cuDNN. The cuDNN library below 6.0 is found to yield a fatal error in batch normalization when handling utterances with long duration in inference.

### Setup

```
sh setup.sh
export LD_LIBRARY_PATH=$PADDLE_INSTALL_DIR/Paddle/third_party/install/warpctc/lib:$LD_LIBRARY_PATH
```

Please replace `$PADDLE_INSTALL_DIR` with your own paddle installation directory.

## Usage

### Preparing Data

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

### Preparing for Training

```
python compute_mean_std.py
```

It will compute mean and stdandard deviation for audio features, and save them to a file with a default name `./mean_std.npz`. This file will be used in both training and inferencing. The default feature of audio data is power spectrum, and the mfcc feature is also supported. To train and infer based on mfcc feature, please generate this file by

```
python compute_mean_std.py --specgram_type mfcc
```

and specify ```--specgram_type mfcc``` when running train.py, infer.py, evaluator.py or tune.py.

More help for arguments:

```
python compute_mean_std.py --help
```

### Training

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

### Preparing language model

The following steps, inference, parameters tuning and evaluating, will require a language model during decoding.
A compressed language model is provided and can be accessed by

```
cd ./lm
sh run.sh
cd ..
```

### Inference

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

### Evaluating

```
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```

More help for arguments:

```
python evaluate.py --help
```

### Parameters tuning

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

### Playing with the ASR Demo

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
