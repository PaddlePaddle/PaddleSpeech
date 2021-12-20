# Tacotron2 with LJSpeech
PaddlePaddle dynamic graph implementation of Tacotron2, a neural network architecture for speech synthesis directly from the text. The implementation is based on [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884).

## Dataset
We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```
## Get Started
Assume the path to the dataset is `~/datasets/LJSpeech-1.1`.
Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize mels.
```bash
./run.sh
```
You can choose a range of stages you want to run, or set `stage` equal to `stop-stage` to use only one stage, for example, running the following command will only preprocess the dataset.
```bash
./run.sh --stage 0 --stop-stage 0
```
### Data Preprocessing
```bash
./local/preprocess.sh ${conf_path}
```
### Model Training
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
Here's the complete help message.
```text
usage: train.py [-h] [--config FILE] [--data DATA_DIR] [--output OUTPUT_DIR]
                [--checkpoint_path CHECKPOINT_PATH] [--ngpu NGPU] [--opts ...]

optional arguments:
  -h, --help            show this help message and exit
  --config FILE         path of the config file to overwrite to default config
                        with.
  --data DATA_DIR       path to the dataset.
  --output OUTPUT_DIR   path to save checkpoint and logs.
  --checkpoint_path CHECKPOINT_PATH
                        path of the checkpoint to load
  --ngpu NGPU           if ngpu == 0, use cpu.
  --opts ...            options to overwrite --config file and the default
                        config, passing in KEY VALUE pairs
```

If you want to train on CPU, just set `--ngpu=0`.
If you want to train on multiple GPUs, just set `--ngpu` as the num of GPU.
By default, training will be resumed from the latest checkpoint in `--output`, if you want to start a new training, please use a new `${OUTPUTPATH}` with no checkpoint.
And if you want to resume from another existing model, you should set `checkpoint_path` to be the checkpoint path you want to load.
**Note: The checkpoint path cannot contain the file extension.**

### Synthesizing
`./local/synthesize.sh` calls `${BIN_DIR}/synthesize.py`,  which synthesize **mels**  from text_list here.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--config FILE] [--checkpoint_path CHECKPOINT_PATH]
                     [--input INPUT] [--output OUTPUT] [--ngpu NGPU]
                     [--opts ...] [-v]

generate mel spectrogram with TransformerTTS.

optional arguments:
  -h, --help            show this help message and exit
  --config FILE         extra config to overwrite the default config
  --checkpoint_path CHECKPOINT_PATH
                        path of the checkpoint to load.
  --input INPUT         path of the text sentences
  --output OUTPUT       path to save outputs
  --ngpu NGPU           if ngpu == 0, use cpu.
  --opts ...            options to overwrite --config file and the default
                        config, passing in KEY VALUE pairs
  -v, --verbose         print msg
```
**Ps.** You can use [waveflow](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc0) as the neural vocoder to synthesize mels to wavs. (Please  refer to `synthesize.sh` in our  LJSpeech waveflow example)

## Pretrained Models
Pretrained Models can be downloaded from the links below. We provide 2 models with different configurations.

1. This model uses a binary classifier to predict the stop token. [tacotron2_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_ljspeech_ckpt_0.3.zip)

2. This model does not have a stop token predictor. It uses the attention peak position to decide whether all the contents have been uttered. Also, guided attention loss is used to speed up training. This model is trained with `configs/alternative.yaml`.[tacotron2_ljspeech_ckpt_0.3_alternative.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_ljspeech_ckpt_0.3_alternative.zip)
