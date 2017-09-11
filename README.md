# DeepSpeech2 on PaddlePaddle

*DeepSpeech2 on PaddlePaddle* is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine, based on [Baidu's Deep Speech 2 paper](http://proceedings.mlr.press/v48/amodei16.pdf), with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform. Our vision is to empower both industrial application and academic research on speech-to-text, via an easy-to-use, efficent and scalable integreted implementation, including training, inferencing & testing module, distributed [PaddleCloud](https://github.com/PaddlePaddle/cloud) training, and demo deployment. Besides, several pre-trained models for both English and Mandarin are also released.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Training a Model](#training-a-model)
- [Data Augmentation Pipeline](#data-augmentation-pipeline)
- [Inference and Evaluation](#inference-and-evaluation)
- [Distributed Cloud Training](#distributed-cloud-training)
- [Hyper-parameters Tuning](#hyper-parameters-tuning)
- [Training for Mandarin Language](#training-for-mandarin-language)
- [Trying Live Demo with Your Own Voice](#trying-live-demo-with-your-own-voice)
- [Experiments and Benchmarks](#experiments-and-benchmarks)
- [Released Models](#released-models)
- [Questions and Help](#questions-and-help)

## Prerequisites
- Only support Python 2.7
- PaddlePaddle the latest version (please refer to the [Installation Guide](https://github.com/PaddlePaddle/Paddle#installation))

## Installation

Please install the [prerequisites](#prerequisites) above before moving on.

```
git clone https://github.com/PaddlePaddle/models.git
cd models/deep_speech_2
sh setup.sh
```

## Getting Started

Several shell scripts provided in `./examples` will help us to quickly give it a try, for most major modules, including data preparation, model training, case inference, model evaluation and demo deployment, with a few public dataset (e.g. [LibriSpeech](http://www.openslr.org/12/), [Aishell](https://github.com/kaldi-asr/kaldi/tree/master/egs/aishell)). Reading these examples will also help us understand how to make it work with our own data.

Some of the scripts in `./examples` are configured with 8 GPUs. If you don't have 8 GPUs available, please modify `CUDA_VISIBLE_DEVICE` and `--trainer_count`. If you don't have any GPU available, please set `--use_gpu` to False to use CPUs instead.

Let's take a tiny sampled subset of [LibriSpeech dataset](http://www.openslr.org/12/) for instance.

- Go to directory

    ```
    cd examples/tiny
    ```

    Notice that this is only a toy example with a tiny sampled subset of LibriSpeech. If we would like to try with the complete dataset (would take several days for training), please go to `examples/librispeech` instead.
- Prepare the data

    ```
    sh run_data.sh
    ```

    `run_data.sh` will download dataset, generate manifests, collect normalizer' statitics and build vocabulary. Once the data preparation is done, we will find the data (only part of LibriSpeech) downloaded in `~/.cache/paddle/dataset/speech/libri` and the corresponding manifest files generated in `./data/tiny` as well as a mean stddev file and a vocabulary file. It has to be run for the very first time we run this dataset and is reusable for all further experiments.
- Train your own ASR model

    ```
    sh run_train.sh
    ```

    `run_train.sh` will start a training job, with training logs printed to stdout and model checkpoint of every pass/epoch saved to `./checkpoints/tiny`. We can resume the training from these checkpoints, or use them for inference, evalutiaton and deployment.
- Case inference with an existing model

    ```
    sh run_infer.sh
    ```

    `run_infer.sh` will show us some speech-to-text decoding results for several (default: 10) samples with the trained model. The performance might not be good now as the current model is only trained with a toy subset of LibriSpeech. To see the results with a better model, we can download a well-trained (trained for several days, with the complete LibriSpeech) model and do the inference:

    ```
    sh run_infer_golden.sh
    ```
- Evaluate an existing model

    ```
    sh run_test.sh
    ```

    `run_test.sh` will evaluate the model with Word Error Rate (or Character Error Rate) measurement. Similarly, we can also download a well-trained model and test its performance:

    ```
    sh run_test_golden.sh
    ```
- Try out a live demo with your own voice

    Until now, we have trained and tested our ASR model qualitatively (`run_infer.sh`) and quantitively (`run_test.sh`) with existing audio files. But we have not yet play the model with our own speech. `demo_server.sh` and `demo_client.sh` helps quickly build up a demo ASR engine with the trained model, enabling us to test and play around with the demo with our own voice.

    We start the server in one console by entering:

    ```
    sh run_demo_server.sh
    ```

    and start the client in another console by entering:

    ```
    sh run_demo_client.sh
    ```

    Then, in the client console, press the `whitespace` key, hold, and start speaking. Until we finish our ulterance, we release the key to let the speech-to-text results show in the console.

    Notice that `run_demo_client.sh` must be run in a machine with a microphone device, while `run_demo_server.sh` could be run in one without any audio recording device, e.g. any remote server. Just be careful to update `run_demo_server.sh` and `run_demo_client.sh` with the actual accessable IP address and port, if the server and client are running with two seperate machines. Nothing has to be done if running in one single machine.

    This demo will first download a pre-trained Mandarin model (trained with 3000 hours of internal speech data). If we would like to try some other model, just update `model_path` argument in the script.  
    
More detailed information are provided in the following sections.

Wish you a happy journey with the DeepSpeech2 ASR engine!


## Data Preparation

#### Generate Manifest

*DeepSpeech2 on PaddlePaddle* accepts a textual **manifest** file as its data set interface. A manifest file summarizes a set of speech data, with each line containing some meta data (e.g. filepath, transcription, duration) of one audio clip, in [JSON](http://www.json.org/) format, such as:

```
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0001.flac", "duration": 3.275, "text": "stuff it into you his belly counselled him"}
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0007.flac", "duration": 4.275, "text": "a cold lucid indifference reigned in his soul"}
```

To use your custom data, you only need to generate such manifest files to summarize the dataset. Given such summarized manifests, training, inference and all other modules can be aware of where to access the audio files, as well as their meta data including the transcription labels.

For how to generate such manifest files, please refer to `data/librispeech/librispeech.py`, which download and generate manifests for LibriSpeech dataset.

#### Compute Mean & Stddev for Normalizer

To perform z-score normalization (zero-mean, unit stddev) upon audio features, we have to estimate in advance the mean and standard deviation of the features, with some training samples:

```
python tools/compute_mean_std.py \
--num_samples 2000 \
--specgram_type linear \
--manifest_paths data/librispeech/manifest.train \
--output_path data/librispeech/mean_std.npz
```

It will compute the mean and standard deviation of power spectgram feature with 2000 random sampled audio clips listed in `data/librispeech/manifest.train` and save the results to `data/librispeech/mean_std.npz` for further usage.


#### Build Vocabulary

A vocabulary of possible characters is required to convert the transcription into a list of token indices for training, and in docoding, to convert from a list of indices back to text again. Such a character-based vocabulary can be build with `tools/build_vocab.py`.

```
python tools/build_vocab.py \
--count_threshold 0 \
--vocab_path data/librispeech/eng_vocab.txt \
--manifest_paths data/librispeech/manifest.train
```

It will write a vocabuary file `data/librispeeech/eng_vocab.txt` with all transcription text in `data/librispeech/manifest.train`, without vocabulary truncation (`--count_threshold 0`).

#### More Help

For more help on arguments:

```
python data/librispeech/librispeech.py --help
python tools/compute_mean_std.py --help
python tools/build_vocab.py --help
```

## Training a model

`train.py` is the main caller of the training module. We show several examples of usage below.

- Start training from scratch with 8 GPUs:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --trainer_count 8
    ```

- Start training from scratch with 16 CPUs:

    ```
    python train.py --use_gpu False --trainer_count 16
    ```
- Resume training from a checkpoint:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --init_model_path CHECKPOINT_PATH_TO_RESUME_FROM
    ```

For more help on arguments:

```
python train.py --help
```
or refer to `example/librispeech/run_train.sh`.

## Data Augmentation Pipeline

Data augmentation has often been a highly effective technique to boost the deep learning performance. We augment our speech data by synthesizing new audios with small random perterbation (label-invariant transformation) added upon raw audios. We don't have to do the syntheses by ourselves, as it is already embeded into the data provider and is done on the fly, randomly for each epoch during training.

Six optional augmentation components are provided for us to configured and inserted into the processing pipeline.

  - Volume Perturbation
  - Speed Perturbation
  - Shifting Perturbation
  - Online Beyesian normalization
  - Noise Perturbation (need background noise audio files)
  - Impulse Response (need impulse audio files)

In order to inform the trainer of what augmentation components we need and what their processing orders are, we are required to prepare a *augmentation configuration file* in [JSON](http://www.json.org/) format. For example:

```
[{
    "type": "speed",
    "params": {"min_speed_rate": 0.95,
               "max_speed_rate": 1.05},
    "prob": 0.6
},
{
    "type": "shift",
    "params": {"min_shift_ms": -5,
               "max_shift_ms": 5},
    "prob": 0.8
}]
```

When the `--augment_conf_file` argument of `trainer.py` is set to the path of the above example configuration file, every audio clip in every epoch will be processed: with 60% of chance, it will first be speed perturbed with a uniformly random sampled speed-rate between 0.95 and 1.05, and then with 80% of chance it will be shifted in time with a random sampled offset between -5 ms and 5 ms. Finally this newly synthesized audio clip will be feed into the feature extractor for further training.

For other configuration examples, please refer to `conf/augmenatation.config.example`.

Be careful when we are utilizing the data augmentation technique, as improper augmentation will do harm to the training, due to the enlarged train-test gap.

## Inference and Evaluation

### Prepare Language Model

A language model is required to improve the decoder's performance. We have prepared two language models (with lossy compression) for users to download and try. One is for English and the other is for Mandarin. Please refer to `models/lm/download_lm_en.sh` and `models/lm/download_lm_zh.sh` for their urls. If you wish to train your own better language model, please refer to [KenLM](https://github.com/kpu/kenlm) for tutorials.

TODO: any other requirements or tips to add?

### Speech-to-text Inference

We provide a inference module `infer.py` to infer, decode and visualize speech-to-text results for several given audio clips. It might help us to have a intuitive and qualitative evaluation of the ASR model's performance.

- Inference with GPU:

    ```
    CUDA_VISIBLE_DEVICES=0 python infer.py --trainer_count 1
    ```

- Inference with CPU:

    ```
    python infer.py --use_gpu False --trainer_count 12
    ```

We provide two types of CTC decoders: *CTC greedy decoder* and *CTC beam search decoder*. The *CTC greedy decoder* is an implementation of the simple best-path decoding algorithm, selecting at each timestep the most likely token, thus being greedy and locally optimal. The [*CTC beam search decoder*](https://arxiv.org/abs/1408.2873) otherwise utilizes a heuristic breadth-first gragh search for reaching a near global optimality; it also requires a pre-trained KenLM language model for better scoring and ranking. The decoder type can be set with argument `--decoding_method`.

For more help on arguments:

```
python infer.py --help
```
or refer to `example/librispeech/run_infer.sh`.

### Evaluate a Model

To evaluate a model's performance quantitively, we can run:

- Evaluation with GPU:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --trainer_count 8
    ```

- Evaluation with CPU:

    ```
    python test.py --use_gpu False --trainer_count 12
    ```

The error rate (default: word error rate; can be set with `--error_rate_type`) will be printed.

For more help on arguments:

```
python test.py --help
```
or refer to `example/librispeech/run_test.sh`.

## Hyper-parameters Tuning

The hyper-parameters $\alpha$ (coefficient for language model scorer) and $\beta$ (coefficient for word count scorer) for the [*CTC beam search decoder*](https://arxiv.org/abs/1408.2873) often have a significant impact on the decoder's performance. It would be better to re-tune them on a validation set when the accustic model is renewed.

`tools/tune.py` performs a 2-D grid search over the hyper-parameter $\alpha$ and $\beta$. We have to provide the range of $\alpha$ and $\beta$, as well as the number of their attempts.

- Tuning with GPU:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/tune.py \
    --trainer_count 8 \
    --alpha_from 0.1 \
    --alpha_to 0.36 \
    --num_alphas 14 \
    --beta_from 0.05 \
    --beta_to 1.0 \
    --num_betas 20
    ```

- Tuning with CPU:

    ```
    python tools/tune.py --use_gpu False
    ```

After tuning, we can reset $\alpha$ and $\beta$ in the inference and evaluation modules to see if they really help improve the ASR performance.

```
python tune.py --help
```
or refer to `example/librispeech/run_tune.sh`.

TODO: add figure.

## Distributed Cloud Training

If you wish to train DeepSpeech2 on PaddleCloud, please refer to
[Train DeepSpeech2 on PaddleCloud](https://github.com/PaddlePaddle/models/tree/develop/deep_speech_2/cloud).

## Training for Mandarin Language

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

## Released Models

## Questions and Help
