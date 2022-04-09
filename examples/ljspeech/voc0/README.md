# WaveFlow with LJSpeech
## Dataset
We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```
## Get Started
Assume the path to the dataset is `~/datasets/LJSpeech-1.1`.
Assume the path to the Tacotron2 generated mels is `../tts0/output/test`.
Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize wavs from mels.
```bash
./run.sh
```
You can choose a range of stages you want to run, or set `stage` equal to `stop-stage` to use only one stage, for example, running the following command will only preprocess the dataset.
```bash
./run.sh --stage 0 --stop-stage 0
```
### Data Preprocessing
```bash
./local/preprocess.sh ${preprocess_path}
```
### Model Training
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path}
```
The training script requires 4 command line arguments.
1. `--data` is the path of the training dataset.
2. `--output` is the path of the output directory.
3. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

If you want distributed training, set a larger `--ngpu` (e.g. 4). Note that distributed training with cpu is not supported yet.

### Synthesizing
`./local/synthesize.sh` calls `${BIN_DIR}/synthesize.py`, which can synthesize waveform from mels.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${input_mel_path} ${train_output_path} ${ckpt_name}
```

Synthesize waveform.
1. We assume the `--input` is a directory containing several mel spectrograms(log magnitude) in `.npy` format.
2. The output would be saved in the `--output` directory, containing several `.wav` files, each with the same name as the mel spectrogram does.
3. `--checkpoint_path` should be the path of the parameter file (`.pdparams`) to load. Note that the extention name `.pdparmas` is not included here.
6. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

## Pretrained Model
Pretrained Model with residual channel equals 128 can be downloaded here:
- [waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/waveflow/waveflow_ljspeech_ckpt_0.3.zip)
