# TransformerTTS with LJSpeech
## Dataset
### Download the datasaet
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
```
### Extract the dataset
```bash
tar xjvf LJSpeech-1.1.tar.bz2
```
## Get Started
Assume the path to the dataset is `~/datasets/LJSpeech-1.1`.
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
│ ├── norm
│ └── raw
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│  ├── norm
│  └── raw
└── train
    ├── norm
    ├── raw
    └── speech_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev` and` test`, each of which contains a `norm` and `raw` sub folder. The raw folder contains speech feature of each utterances, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/speech_stats.npy`.

Also there is a `metadata.jsonl` in each subfolder. It is a table-like file which contains phones, text_lengths, speech_lengths, path of speech features, speaker and id of each utterance.

### Train the model
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
Here's the complete help message.
```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--device DEVICE] [--nprocs NPROCS] [--verbose VERBOSE]
                [--phones-dict PHONES_DICT]

Train a TransformerTTS model with LJSpeech TTS dataset.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file to overwrite default config.
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
```
1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are save in `checkpoints/` inside this directory.
4. `--device` is the type of the device to run the experiment, 'cpu' or 'gpu' are supported.
5. `--nprocs` is the number of processes to run in parallel, note that nprocs > 1 is only supported when `--device` is 'gpu'.
6. `--phones-dict` is the path of the phone vocabulary file.

## Synthesize
We use [waveflow](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/ljspeech/voc0) as the neural vocoder.
Download Pretrained WaveFlow Model with residual channel equals 128 from [waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip) and unzip it.
```bash
unzip waveflow_ljspeech_ckpt_0.3.zip
```
WaveFlow  checkpoint contains files listed below.
```text
waveflow_ljspeech_ckpt_0.3
├── config.yaml           # default config used to train waveflow
└── step-2000000.pdparams # model parameters of waveflow
```
`./local/synthesize.sh` calls `${BIN_DIR}/synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--transformer-tts-config TRANSFORMER_TTS_CONFIG]
                     [--transformer-tts-checkpoint TRANSFORMER_TTS_CHECKPOINT]
                     [--transformer-tts-stat TRANSFORMER_TTS_STAT]
                     [--waveflow-config WAVEFLOW_CONFIG]
                     [--waveflow-checkpoint WAVEFLOW_CHECKPOINT]
                     [--phones-dict PHONES_DICT]
                     [--test-metadata TEST_METADATA] [--output-dir OUTPUT_DIR]
                     [--device DEVICE] [--verbose VERBOSE]

Synthesize with transformer tts & waveflow.

optional arguments:
  -h, --help            show this help message and exit
  --transformer-tts-config TRANSFORMER_TTS_CONFIG
                        transformer tts config file.
  --transformer-tts-checkpoint TRANSFORMER_TTS_CHECKPOINT
                        transformer tts checkpoint to load.
  --transformer-tts-stat TRANSFORMER_TTS_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training transformer tts.
  --waveflow-config WAVEFLOW_CONFIG
                        waveflow config file.
  --waveflow-checkpoint WAVEFLOW_CHECKPOINT
                        waveflow checkpoint to load.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
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
usage: synthesize_e2e.py [-h]
                         [--transformer-tts-config TRANSFORMER_TTS_CONFIG]
                         [--transformer-tts-checkpoint TRANSFORMER_TTS_CHECKPOINT]
                         [--transformer-tts-stat TRANSFORMER_TTS_STAT]
                         [--waveflow-config WAVEFLOW_CONFIG]
                         [--waveflow-checkpoint WAVEFLOW_CHECKPOINT]
                         [--phones-dict PHONES_DICT] [--text TEXT]
                         [--output-dir OUTPUT_DIR] [--device DEVICE]
                         [--verbose VERBOSE]

Synthesize with transformer tts & waveflow.

optional arguments:
  -h, --help            show this help message and exit
  --transformer-tts-config TRANSFORMER_TTS_CONFIG
                        transformer tts config file.
  --transformer-tts-checkpoint TRANSFORMER_TTS_CHECKPOINT
                        transformer tts checkpoint to load.
  --transformer-tts-stat TRANSFORMER_TTS_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training transformer tts.
  --waveflow-config WAVEFLOW_CONFIG
                        waveflow config file.
  --waveflow-checkpoint WAVEFLOW_CHECKPOINT
                        waveflow checkpoint to load.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line.
  --output-dir OUTPUT_DIR
                        output dir.
  --device DEVICE       device type to use.
  --verbose VERBOSE     verbose.
```
1. `--transformer-tts-config`, `--transformer-tts-checkpoint`, `--transformer-tts-stat` and `--phones-dict` are arguments for transformer_tts, which correspond to the 4 files in the transformer_tts pretrained model.
2. `--waveflow-config`, `--waveflow-checkpoint` are arguments for waveflow, which correspond to the 2 files in the waveflow pretrained model.
3. `--test-metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
4. `--text` is the text file, which contains sentences to synthesize.
5. `--output-dir` is the directory to save synthesized audio files.
6. `--device` is the type of device to run synthesis, 'cpu' and 'gpu' are supported. 'gpu' is recommended for faster synthesis.

## Pretrained Model
Pretrained Model can be downloaded here. [transformer_tts_ljspeech_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.4.zip)

TransformerTTS  checkpoint contains files listed below.
```text
transformer_tts_ljspeech_ckpt_0.4
├── default.yaml             # default config used to train transformer_tts
├── phone_id_map.txt         # phone vocabulary file when training transformer_tts
├── snapshot_iter_201500.pdz # model parameters and optimizer states
└── speech_stats.npy         # statistics used to normalize spectrogram when training transformer_tts
```
You can use the following scripts to synthesize for `${BIN_DIR}/../sentences_en.txt` using pretrained transformer_tts  and waveflow models.
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
  --transformer-tts-config=transformer_tts_ljspeech_ckpt_0.4/default.yaml \
  --transformer-tts-checkpoint=transformer_tts_ljspeech_ckpt_0.4/snapshot_iter_201500.pdz \
  --transformer-tts-stat=transformer_tts_ljspeech_ckpt_0.4/speech_stats.npy \
  --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
  --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
  --text=${BIN_DIR}/../sentences_en.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=transformer_tts_ljspeech_ckpt_0.4/phone_id_map.txt
```
