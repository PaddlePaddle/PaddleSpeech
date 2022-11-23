# Style MelGAN with CSMSC
This example contains code used to train a [Style MelGAN](https://arxiv.org/abs/2011.01557) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).
## Dataset
### Download and Extract
Download CSMSC from it's [official website](https://test.data-baker.com/data/index/TNtts/) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/BZNSYP`.

### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) results to cut the silence in the edge of audio.
You can download from here [baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) of our repo.

## Get Started
Assume the path to the dataset is `~/datasets/BZNSYP`.
Assume the path to the MFA result of CSMSC is `./baker_alignment_tone`.
Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize wavs.
    - synthesize waveform from `metadata.jsonl`.
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
When it is done. A `dump` folder is created in the current directory. The structure of the dump folder is listed below.

```text
dump
├── dev
│   ├── norm
│   └── raw
├── test
│   ├── norm
│   └── raw
└── train
    ├── norm
    ├── raw
    └── feats_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev`, and `test`, each of which contains a `norm` and `raw` subfolder. The `raw` folder contains the log magnitude of the mel spectrogram of each utterance, while the norm folder contains the normalized spectrogram. The statistics used to normalize the spectrogram are computed from the training set, which is located in `dump/train/feats_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains id and paths to the spectrogram of each utterance.

### Model Training
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` calls `${BIN_DIR}/train.py`.
Here's the complete help message.

```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU]

Train a Style MelGAN model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Style MelGAN config file.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.
```

1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are saved in `checkpoints/` inside this directory.
4. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

### Synthesizing
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--generator-type GENERATOR_TYPE] [--config CONFIG]
                     [--checkpoint CHECKPOINT] [--test-metadata TEST_METADATA]
                     [--output-dir OUTPUT_DIR] [--ngpu NGPU]

Synthesize with GANVocoder.

optional arguments:
  -h, --help            show this help message and exit
  --generator-type GENERATOR_TYPE
                        type of GANVocoder, should in {pwgan, mb_melgan,
                        style_melgan, } now
  --config CONFIG       GANVocoder config file.
  --checkpoint CHECKPOINT
                        snapshot to load.
  --test-metadata TEST_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.
```

1. `--config` style melgan config file. You should use the same config with which the model is trained.
2. `--checkpoint` is the checkpoint to load. Pick one of the checkpoints from `checkpoints` inside the training output directory.
3. `--test-metadata` is the metadata of the test dataset. Use the `metadata.jsonl` in the `dev/norm` subfolder from the processed directory.
4. `--output-dir` is the directory to save the synthesized audio files.
5. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

## Pretrained Models
The pretrained model can be downloaded here:
- [style_melgan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/style_melgan/style_melgan_csmsc_ckpt_0.1.1.zip)

The static model of Style MelGAN is not available now.

Style MelGAN checkpoint contains files listed below.

```text
hifigan_csmsc_ckpt_0.1.1
├── default.yaml                    # default config used to train style melgan
├── feats_stats.npy                  # statistics used to normalize spectrogram when training style melgan 
└── snapshot_iter_1500000.pdz     # generator parameters of style melgan
```

## Acknowledgement
We adapted some code from https://github.com/kan-bayashi/ParallelWaveGAN.
