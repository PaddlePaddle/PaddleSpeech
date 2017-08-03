# Deep Speech 2 on PaddlePaddle

## Installation

### Prerequisites

 - **Python = 2.7** only supported;
 - **cuDNN >= 6.0** is required to utilize NVIDIA GPU platform in the installation of PaddlePaddle, and the **CUDA toolkit** with proper version suitable for cuDNN. The cuDNN library below 6.0 is found to yield a fatal error in batch normalization when handling utterances with long duration in inference.

### Setup for Training & Evaluation

```
sh setup.sh
export LD_LIBRARY_PATH=$PADDLE_INSTALL_DIR/Paddle/third_party/install/warpctc/lib:$LD_LIBRARY_PATH
```

Please replace `$PADDLE_INSTALL_DIR` with your own paddle installation directory.

### Setup for Demo

Please do the following extra installation before run `demo_client.py` to try the realtime ASR demo. However there is no need to install them for the computer running the demo's server-end (`demo_server.py`). For details of running the ASR demo, please refer to the [section](#playing-with-the-asr-demo).

For example, on MAC OS X:

```
brew install portaudio
pip install pyaudio
pip install pynput
```


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

A real-time ASR demo (`demo_server.py` and `demo_client.py`) are prepared for users to try out the ASR model with their own voice. After a model and language model is prepared, we can first start the demo server:

```
CUDA_VISIBLE_DEVICES=0 python demo_server.py
```
And then in another console, start the client:

```
python demo_client.py
```
On the client console, press and hold "white-space" key and start talking, then release the "white-space" key when you finish your speech. The decoding results (infered transcription) will be displayed.

If you would like to start the server and the client in two machines. Please use `--host_ip` and `--host_port` to indicate the actual IP address and port, for both `demo_server.py` and `demo_client.py`.

Notice that `demo_client.py` should be started in your local computer with microphone hardware, while `demo_server.py` can be started in any remote server as well as the same local computer. IP address and port should be properly set for server-client communication.

For running `demo_client.py`, please first finish the [extra installation steps](#setup-for-demo).
