# Speedyspeech with CSMSC
This example contains code used to train a [Speedyspeech](http://arxiv.org/abs/2008.03802) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html). NOTE that we only implement the student part of the Speedyspeech model. The ground truth alignment used to train the model is extracted from the dataset using [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).

## Dataset
### Download and Extract the datasaet
Download CSMSC from it's [Official Website](https://test.data-baker.com/data/index/source).

### Get MFA result of CSMSC and Extract it
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for SPEEDYSPEECH.
You can download from here [baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz), or train your own MFA model reference to  [use_mfa example](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/other/use_mfa) of our repo.

## Get Started
Assume the path to the dataset is `~/datasets/BZNSYP`.
Assume the path to the MFA result of CSMSC is `./baker_alignment_tone`.
Run the command below to
1. **source path**.
2. preprocess the dataset,
3. train the model.
4. synthesize wavs.
    - synthesize waveform from `metadata.jsonl`.
    - synthesize waveform from text file.
6. inference using static model.
```bash
./run.sh
```
### Preprocess the dataset
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

The dataset is split into 3 parts, namely `train`, `dev` and `test`, each of which contains a `norm` and `raw` sub folder. The raw folder contains log magnitude of mel spectrogram of each utterances, while the norm folder contains normalized spectrogram. The statistics used to normalize the spectrogram is computed from the training set, which is located in `dump/train/feats_stats.npy`.

Also there is a `metadata.jsonl` in each subfolder. It is a table-like file which contains phones, tones, durations, path of spectrogram, and id of each utterance.

### Train the model
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
```
Here's the complete help message.
```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                     [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                     [--device DEVICE] [--nprocs NPROCS] [--verbose VERBOSE]
                     [--use-relative-path USE_RELATIVE_PATH]
                     [--phones-dict PHONES_DICT] [--tones-dict TONES_DICT]

Train a Speedyspeech model with sigle speaker dataset.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --device DEVICE       device type to use.
  --nprocs NPROCS       number of processes.
  --verbose VERBOSE     verbose.
  --use-relative-path USE_RELATIVE_PATH
                        whether use relative path in metadata
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --tones-dict TONES_DICT
                        tone vocabulary file.
```

1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are save in `checkpoints/` inside this directory.
4. `--device` is the type of the device to run the experiment, 'cpu' or 'gpu' are supported.
5. `--nprocs` is the number of processes to run in parallel, note that nprocs > 1 is only supported when `--device` is 'gpu'.
6. `--phones-dict` is the path of the phone vocabulary file.
7. `--tones-dict` is the path of the tone vocabulary file.

### Synthesize
We use [parallel wavegan](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/csmsc/voc1) as the neural vocoder.
Download pretrained parallel wavegan model from [pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip) and unzip it.
```bash
unzip pwg_baker_ckpt_0.4.zip
```
Parallel WaveGAN checkpoint contains files listed below.
```text
pwg_baker_ckpt_0.4
├── pwg_default.yaml               # default config used to train parallel wavegan
├── pwg_snapshot_iter_400000.pdz   # model parameters of parallel wavegan
└── pwg_stats.npy                  # statistics used to normalize spectrogram when training parallel wavegan
```
`./local/synthesize.sh` calls `${BIN_DIR}/synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--speedyspeech-config SPEEDYSPEECH_CONFIG]
                     [--speedyspeech-checkpoint SPEEDYSPEECH_CHECKPOINT]
                     [--speedyspeech-stat SPEEDYSPEECH_STAT]
                     [--pwg-config PWG_CONFIG]
                     [--pwg-checkpoint PWG_CHECKPOINT] [--pwg-stat PWG_STAT]
                     [--phones-dict PHONES_DICT] [--tones-dict TONES_DICT]
                     [--test-metadata TEST_METADATA] [--output-dir OUTPUT_DIR]
                     [--inference-dir INFERENCE_DIR] [--device DEVICE]
                     [--verbose VERBOSE]

Synthesize with speedyspeech & parallel wavegan.

optional arguments:
  -h, --help            show this help message and exit
  --speedyspeech-config SPEEDYSPEECH_CONFIG
                        config file for speedyspeech.
  --speedyspeech-checkpoint SPEEDYSPEECH_CHECKPOINT
                        speedyspeech checkpoint to load.
  --speedyspeech-stat SPEEDYSPEECH_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training speedyspeech.
  --pwg-config PWG_CONFIG
                        config file for parallelwavegan.
  --pwg-checkpoint PWG_CHECKPOINT
                        parallel wavegan generator parameters to load.
  --pwg-stat PWG_STAT   mean and standard deviation used to normalize
                        spectrogram when training speedyspeech.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --tones-dict TONES_DICT
                        tone vocabulary file.
  --test-metadata TEST_METADATA
                        test metadata
  --output-dir OUTPUT_DIR
                        output dir
  --inference-dir INFERENCE_DIR
                        dir to save inference models
  --device DEVICE       device type to use
  --verbose VERBOSE     verbose
```
`./local/synthesize_e2e.sh` calls `${BIN_DIR}/synthesize_e2e.py`, which can synthesize waveform from text file.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h] [--speedyspeech-config SPEEDYSPEECH_CONFIG]
                         [--speedyspeech-checkpoint SPEEDYSPEECH_CHECKPOINT]
                         [--speedyspeech-stat SPEEDYSPEECH_STAT]
                         [--pwg-config PWG_CONFIG]
                         [--pwg-checkpoint PWG_CHECKPOINT]
                         [--pwg-stat PWG_STAT] [--text TEXT]
                         [--phones-dict PHONES_DICT] [--tones-dict TONES_DICT]
                         [--output-dir OUTPUT_DIR]
                         [--inference-dir INFERENCE_DIR] [--device DEVICE]
                         [--verbose VERBOSE]

Synthesize with speedyspeech & parallel wavegan.

optional arguments:
  -h, --help            show this help message and exit
  --speedyspeech-config SPEEDYSPEECH_CONFIG
                        config file for speedyspeech.
  --speedyspeech-checkpoint SPEEDYSPEECH_CHECKPOINT
                        speedyspeech checkpoint to load.
  --speedyspeech-stat SPEEDYSPEECH_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training speedyspeech.
  --pwg-config PWG_CONFIG
                        config file for parallelwavegan.
  --pwg-checkpoint PWG_CHECKPOINT
                        parallel wavegan checkpoint to load.
  --pwg-stat PWG_STAT   mean and standard deviation used to normalize
                        spectrogram when training speedyspeech.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --tones-dict TONES_DICT
                        tone vocabulary file.
  --output-dir OUTPUT_DIR
                        output dir
  --inference-dir INFERENCE_DIR
                        dir to save inference models
  --device DEVICE       device type to use
  --verbose VERBOSE     verbose
```
1. `--speedyspeech-config`, `--speedyspeech-checkpoint`, `--speedyspeech-stat` are arguments for speedyspeech, which correspond to the 3 files in the speedyspeech pretrained model.
2. `--pwg-config`, `--pwg-checkpoint`, `--pwg-stat` are arguments for parallel wavegan, which correspond to the 3 files in the parallel wavegan pretrained model.
3. `--text` is the text file, which contains sentences to synthesize.
4. `--output-dir` is the directory to save synthesized audio files.
5. `--inference-dir` is the directory to save exported model, which can be used with paddle infernece.
6. `--device` is the type of device to run synthesis, 'cpu' and 'gpu' are supported. 'gpu' is recommended for faster synthesis.
7. `--phones-dict` is the path of the phone vocabulary file.
8. `--tones-dict` is the path of the tone vocabulary file.

### Inference
After Synthesize, we will get static models of speedyspeech and pwgan in `${train_output_path}/inference`.
`./local/inference.sh` calls `${BIN_DIR}/inference.py`, which provides a paddle static model inference example for speedyspeech + pwgan synthesize.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path}
```

## Pretrained Model
Pretrained SpeedySpeech model with no silence in the edge of audios. [speedyspeech_nosil_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/speedyspeech_nosil_baker_ckpt_0.5.zip)

SpeedySpeech checkpoint contains files listed below.
```text
speedyspeech_nosil_baker_ckpt_0.5
├── default.yaml            # default config used to train speedyspeech
├── feats_stats.npy         # statistics used to normalize spectrogram when training speedyspeech
├── phone_id_map.txt        # phone vocabulary file when training speedyspeech
├── snapshot_iter_11400.pdz # model parameters and optimizer states
└── tone_id_map.txt         # tone vocabulary file when training speedyspeech
```
You can use the following scripts to synthesize for `${BIN_DIR}/../sentences.txt` using pretrained speedyspeech and parallel wavegan models.
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
  --speedyspeech-config=speedyspeech_nosil_baker_ckpt_0.5/default.yaml \
  --speedyspeech-checkpoint=speedyspeech_nosil_baker_ckpt_0.5/snapshot_iter_11400.pdz \
  --speedyspeech-stat=speedyspeech_nosil_baker_ckpt_0.5/feats_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --text=${BIN_DIR}/../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --inference-dir=exp/default/inference \
  --device="gpu" \
  --phones-dict=speedyspeech_nosil_baker_ckpt_0.5/phone_id_map.txt \
  --tones-dict=speedyspeech_nosil_baker_ckpt_0.5/tone_id_map.txt
```
