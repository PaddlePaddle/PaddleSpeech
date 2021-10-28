# FastSpeech2 with CSMSC
This example contains code used to train a [Fastspeech2](https://arxiv.org/abs/2006.04558) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).

## Dataset
### Download and Extract the datasaet
Download CSMSC from it's [Official Website](https://test.data-baker.com/data/index/source).

### Get MFA result of CSMSC and Extract it
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2.
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
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └── raw
└── train
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev` and` test`, each of which contains a `norm` and `raw` sub folder. The raw folder contains speech、pitch and energy features of each utterances, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`.

Also there is a `metadata.jsonl` in each subfolder. It is a table-like file which contains phones, text_lengths, speech_lengths, durations, path of speech features, path of pitch features, path of energy features, speaker and id of each utterance.

### Train the model
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` calls `${BIN_DIR}/train.py`.
Here's the complete help message.
```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--device DEVICE] [--nprocs NPROCS] [--verbose VERBOSE]
                [--phones-dict PHONES_DICT] [--speaker-dict SPEAKER_DICT]

Train a FastSpeech2 model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       fastspeech2 config file.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --device DEVICE       device type to use.
  --nprocs NPROCS       number of processes.
  --verbose VERBOSE     verbose.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --speaker-dict SPEAKER_DICT
                        speaker id map file for multiple speaker model.
```
1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are save in `checkpoints/` inside this directory.
4. `--device` is the type of the device to run the experiment, 'cpu' or 'gpu' are supported.
5. `--nprocs` is the number of processes to run in parallel, note that nprocs > 1 is only supported when `--device` is 'gpu'.
6. `--phones-dict` is the path of the phone vocabulary file.

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
usage: synthesize.py [-h] [--fastspeech2-config FASTSPEECH2_CONFIG]
                     [--fastspeech2-checkpoint FASTSPEECH2_CHECKPOINT]
                     [--fastspeech2-stat FASTSPEECH2_STAT]
                     [--pwg-config PWG_CONFIG]
                     [--pwg-checkpoint PWG_CHECKPOINT] [--pwg-stat PWG_STAT]
                     [--phones-dict PHONES_DICT] [--speaker-dict SPEAKER_DICT]
                     [--test-metadata TEST_METADATA] [--output-dir OUTPUT_DIR]
                     [--device DEVICE] [--verbose VERBOSE]

Synthesize with fastspeech2 & parallel wavegan.

optional arguments:
  -h, --help            show this help message and exit
  --fastspeech2-config FASTSPEECH2_CONFIG
                        fastspeech2 config file.
  --fastspeech2-checkpoint FASTSPEECH2_CHECKPOINT
                        fastspeech2 checkpoint to load.
  --fastspeech2-stat FASTSPEECH2_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training fastspeech2.
  --pwg-config PWG_CONFIG
                        parallel wavegan config file.
  --pwg-checkpoint PWG_CHECKPOINT
                        parallel wavegan generator parameters to load.
  --pwg-stat PWG_STAT   mean and standard deviation used to normalize
                        spectrogram when training parallel wavegan.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --speaker-dict SPEAKER_DICT
                        speaker id map file for multiple speaker model.
  --test-metadata TEST_METADATA
                        test metadata.
  --output-dir OUTPUT_DIR
                        output dir.
  --device DEVICE       device type to use.
  --verbose VERBOSE     verbose.
```
`./local/synthesize_e2e.sh` calls `${BIN_DIR}/synthesize_e2e.py`, which can synthesize waveform from text file.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h] [--fastspeech2-config FASTSPEECH2_CONFIG]
                         [--fastspeech2-checkpoint FASTSPEECH2_CHECKPOINT]
                         [--fastspeech2-stat FASTSPEECH2_STAT]
                         [--pwg-config PWG_CONFIG]
                         [--pwg-checkpoint PWG_CHECKPOINT]
                         [--pwg-stat PWG_STAT] [--phones-dict PHONES_DICT]
                         [--text TEXT] [--output-dir OUTPUT_DIR]
                         [--device DEVICE] [--verbose VERBOSE]

Synthesize with fastspeech2 & parallel wavegan.

optional arguments:
  -h, --help            show this help message and exit
  --fastspeech2-config FASTSPEECH2_CONFIG
                        fastspeech2 config file.
  --fastspeech2-checkpoint FASTSPEECH2_CHECKPOINT
                        fastspeech2 checkpoint to load.
  --fastspeech2-stat FASTSPEECH2_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training fastspeech2.
  --pwg-config PWG_CONFIG
                        parallel wavegan config file.
  --pwg-checkpoint PWG_CHECKPOINT
                        parallel wavegan generator parameters to load.
  --pwg-stat PWG_STAT   mean and standard deviation used to normalize
                        spectrogram when training parallel wavegan.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line.
  --output-dir OUTPUT_DIR
                        output dir.
  --device DEVICE       device type to use.
  --verbose VERBOSE     verbose.
```

1. `--fastspeech2-config`, `--fastspeech2-checkpoint`, `--fastspeech2-stat` and `--phones-dict` are arguments for fastspeech2, which correspond to the 4 files in the fastspeech2 pretrained model.
2. `--pwg-config`, `--pwg-checkpoint`, `--pwg-stat` are arguments for parallel wavegan, which correspond to the 3 files in the parallel wavegan pretrained model.
3. `--test-metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
4. `--text` is the text file, which contains sentences to synthesize.
5. `--output-dir` is the directory to save synthesized audio files.
6. `--device is` the type of device to run synthesis, 'cpu' and 'gpu' are supported. 'gpu' is recommended for faster synthesis.

## Pretrained Model
Pretrained FastSpeech2 model with no silence in the edge of audios. [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_baker_ckpt_0.4.zip)

FastSpeech2 checkpoint contains files listed below.
```text
fastspeech2_nosil_baker_ckpt_0.4
├── default.yaml            # default config used to train fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── snapshot_iter_76000.pdz # model parameters and optimizer states
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```
You can use the following scripts to synthesize for `${BIN_DIR}/../sentences.txt` using pretrained fastspeech2 and parallel wavegan models.
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
  --fastspeech2-config=fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
  --fastspeech2-checkpoint=fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
  --fastspeech2-stat=fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --text=${BIN_DIR}/../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
```
