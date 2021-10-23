# Tacotron2

PaddlePaddle dynamic graph implementation of Tacotron2, a neural network architecture for speech synthesis directly from text. The implementation is based on [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884).

## Project Structure

```text
├── config.py              # default configuration file
├── ljspeech.py            # dataset and dataloader settings for LJSpeech
├── preprocess.py          # script to preprocess LJSpeech dataset
├── synthesize.py          # script to synthesize spectrogram from text
├── train.py               # script for tacotron2 model training
├── synthesize.ipynb       # notebook example for end-to-end TTS
```

## Dataset

We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

Then you need to preprocess the data by running ``preprocess.py``, the preprocessed data will be placed in ``--output`` directory.

```bash
python preprocess.py \
--input=${DATAPATH} \
--output=${PREPROCESSEDDATAPATH} \
-v  \
```

For more help on arguments

``python preprocess.py --help``.

## Train the model

Tacotron2 model can be trained by running ``train.py``.

```bash
python train.py \
--data=${PREPROCESSEDDATAPATH} \
--output=${OUTPUTPATH} \
--device=gpu \
```

If you want to train on CPU, just set ``--device=cpu``.
If you want to train on multiple GPUs, just set ``--nprocs`` as num of GPU.
By default, training will be resumed from the latest checkpoint in ``--output``, if you want to start a new training, please use a new ``${OUTPUTPATH}`` with no checkpoint. And if you want to resume from an other existing model, you should set ``checkpoint_path`` to be the checkpoint path you want to load.

**Note: The checkpoint path cannot contain the file extension.**

For more help on arguments

``python train_transformer.py --help``.

## Synthesize

After training the Tacotron2, spectrogram can be synthesized by running ``synthesize.py``.

```bash
python synthesize.py \
--config=${CONFIGPATH} \
--checkpoint_path=${CHECKPOINTPATH} \
--input=${TEXTPATH} \
--output=${OUTPUTPATH}
--device=gpu
```

The ``${CONFIGPATH}`` needs to be matched with ``${CHECKPOINTPATH}``.

For more help on arguments

``python synthesize.py --help``.

Then you can find the spectrogram files in ``${OUTPUTPATH}``, and then they can be the input of vocoder like [waveflow](../waveflow/README.md#Synthesis) to get audio files.


## Pretrained Models

Pretrained Models can be downloaded from links below. We provide 2 models with different configurations.

1. This model use a binary classifier to predict the stop token. [tacotron2_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3.zip)

2. This model does not have a stop token predictor. It uses the attention peak position to decided whether all the contents have been uttered. Also guided attention loss is used to speed up training. This model is trained with `configs/alternative.yaml`.[tacotron2_ljspeech_ckpt_0.3_alternative.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3_alternative.zip)


## Notebook: End-to-end TTS

See [synthesize.ipynb](./synthesize.ipynb) for details about end-to-end TTS with tacotron2 and waveflow.
