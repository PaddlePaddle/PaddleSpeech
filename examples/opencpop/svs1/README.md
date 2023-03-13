([简体中文](./README_cn.md)|English)
# DiffSinger with Opencpop
This example contains code used to train a [DiffSinger](https://arxiv.org/abs/2105.02446) model with [Mandarin singing corpus](https://wenet.org.cn/opencpop/).

## Dataset
### Download and Extract
Download Opencpop from it's [Official Website](https://wenet.org.cn/opencpop/download/) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/Opencpop`.

## Get Started
Assume the path to the dataset is `~/datasets/Opencpop`.
Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize wavs.
    - synthesize waveform from `metadata.jsonl`.
    - (Supporting) synthesize waveform from a text file. 
5. (Supporting) inference using the static model.
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
    ├── speech_stats.npy
    └── speech_stretchs.npy

```
The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains speech, pitch and energy features of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`. `speech_stretchs.npy` contains the minimum and maximum values of each dimension of the mel spectrum, which is used for linear stretching before training/inference of the diffusion module.
Note: Since the training effect of non-norm features is due to norm, the features saved under `norm` are features that have not been normed.


Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains utterance id, speaker id, phones, text_lengths, speech_lengths, phone durations, the path of speech features, the path of pitch features, the path of energy features, note, note durations, slur.

### Model Training
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` calls `${BIN_DIR}/train.py`.
Here's the complete help message.
```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU] [--phones-dict PHONES_DICT]
                [--speaker-dict SPEAKER_DICT] [--speech-stretchs SPEECH_STRETCHS]

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
  --ngpu NGPU           if ngpu=0, use cpu.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --speaker-dict SPEAKER_DICT
                        speaker id map file for multiple speaker model.
  --speech-stretchs SPEECH_STRETCHS
                        min amd max mel for stretching.
```
1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are saved in `checkpoints/` inside this directory.
4. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.
5. `--phones-dict` is the path of the phone vocabulary file.
6. `--speech-stretchs` is the path of mel's min-max data file.

### Synthesizing
We use parallel wavegan as the neural vocoder.
Download pretrained parallel wavegan model from [pwgan_opencpop_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/t2s/svs/opencpop/pwgan_opencpop_ckpt_1.4.0.zip) and unzip it.
```bash
unzip pwgan_opencpop_ckpt_1.4.0.zip
```
Parallel WaveGAN checkpoint contains files listed below.
```text
pwgan_opencpop_ckpt_1.4.0.zip
├── default.yaml                   # default config used to train parallel wavegan
├── snapshot_iter_100000.pdz       # model parameters of parallel wavegan
└── feats_stats.npy                # statistics used to normalize spectrogram when training parallel wavegan
```
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h]
                     [--am {diffsinger_opencpop}]
                     [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                     [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                     [--voc {pwgan_opencpop}]
                     [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                     [--voc_stat VOC_STAT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]
                     [--speech_stretchs SPEECH_STRETCHS]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech,tacotron2_aishell3}
                        Choose acoustic model type of tts task.
  --am_config AM_CONFIG
                        Config of acoustic model.
  --am_ckpt AM_CKPT     Checkpoint file of acoustic model.
  --am_stat AM_STAT     mean and standard deviation used to normalize
                        spectrogram when training acoustic model.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --tones_dict TONES_DICT
                        tone vocabulary file.
  --speaker_dict SPEAKER_DICT
                        speaker id map file.
  --voice-cloning VOICE_CLONING
                        whether training voice cloning model.
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,wavernn_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,style_melgan_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --test_metadata TEST_METADATA
                        test metadata.
  --output_dir OUTPUT_DIR
                        output dir.
  --speech-stretchs     mel min and max values file.
```


## Pretrained Model
Pretrained DiffSinger model:
- [diffsinger_opencpop_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/t2s/svs/opencpop/diffsinger_opencpop_ckpt_1.4.0.zip)

DiffSinger checkpoint contains files listed below.
```text
diffsinger_opencpop_ckpt_1.4.0.zip
├── default.yaml             # default config used to train diffsinger
├── energy_stats.npy         # statistics used to normalize energy when training diffsinger if norm is needed
├── phone_id_map.txt         # phone vocabulary file when training diffsinger
├── pitch_stats.npy          # statistics used to normalize pitch when training diffsinger if norm is needed 
├── snapshot_iter_160000.pdz # model parameters of diffsinger
├── speech_stats.npy         # statistics used to normalize mel when training diffsinger if norm is needed
└── speech_stretchs.npy      # Min and max values to use for mel spectral stretching before training diffusion

```
At present, the text frontend is not perfect, and the method of `synthesize_e2e` is not supported for synthesizing audio. Try using `synthesize` first.