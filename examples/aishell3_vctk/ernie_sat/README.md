# ERNIE-SAT with AISHELL3 and VCTK dataset

ERNIE-SAT 是可以同时处理中英文的跨语言的语音-语言跨模态大模型，其在语音编辑、个性化语音合成以及跨语言的语音合成等多个任务取得了领先效果。可以应用于语音编辑、个性化合成、语音克隆、同传翻译等一系列场景，该项目供研究使用。

## 模型框架
ERNIE-SAT 中我们提出了两项创新：
- 在预训练过程中将中英双语对应的音素作为输入，实现了跨语言、个性化的软音素映射
- 采用语言和语音的联合掩码学习实现了语言和语音的对齐

<p align="center">
    <img src="https://user-images.githubusercontent.com/24568452/186110814-1b9c6618-a0ab-4c0c-bb3d-3d860b0e8cc2.png" />
</p>

## Dataset
### Download and Extract
Download all datasets and extract it to `~/datasets`:
- The aishell3 dataset is in the directory `~/datasets/data_aishell3`
- The vctk dataset is in the directory `~/datasets/VCTK-Corpus-0.92`
 
### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for the fastspeech2 training.
You can download from here:
- [aishell3_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz) 
- [vctk_alignment.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/VCTK-Corpus-0.92/vctk_alignment.tar.gz)

Or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) (use MFA1.x now) of our repo.

## Get Started
Assume the paths to the datasets are:
- `~/datasets/data_aishell3` 
- `~/datasets/VCTK-Corpus-0.92`

Assume the path to the MFA results of the datasets are:
- `./aishell3_alignment_tone`
- `./vctk_alignment`

Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize wavs.
    - synthesize waveform from `metadata.jsonl`.
    - synthesize waveform from text file.

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
    ├── norm
    ├── raw
    └── speech_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains speech features of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, speech_lengths, durations, the path of speech features, speaker, and id of each utterance.

### Model Training
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` calls `${BIN_DIR}/train.py`.

### Synthesizing
We use [HiFiGAN](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc5) as the neural vocoder.

Download pretrained HiFiGAN model from [hifigan_aishell3_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip) and unzip it.
```bash
unzip hifigan_aishell3_ckpt_0.2.0.zip
```
HiFiGAN checkpoint contains files listed below.
```text
hifigan_aishell3_ckpt_0.2.0
├── default.yaml                    # default config used to train HiFiGAN
├── feats_stats.npy                 # statistics used to normalize spectrogram when training HiFiGAN
└── snapshot_iter_2500000.pdz       # generator parameters of HiFiGAN
```
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
##  Speech Synthesis and Speech Editing
### Prepare

**prepare aligner**
```bash
mkdir -p tools/aligner
cd tools
# download MFA
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
# extract MFA
tar xvf montreal-forced-aligner_linux.tar.gz
# fix .so of MFA
cd montreal-forced-aligner/lib
ln -snf libpython3.6m.so.1.0 libpython3.6m.so
cd -
# download align models and dicts
cd aligner
wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/aishell3_model.zip
wget https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/simple.lexicon
wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/vctk_model.zip
wget https://paddlespeech.bj.bcebos.com/MFA/LJSpeech-1.1/cmudict-0.7b
cd ../../
```
**prepare pretrained FastSpeech2 models**

ERNIE-SAT use FastSpeech2 as phoneme duration predictor:
```bash
mkdir download
cd download
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_baker_ckpt_0.5.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip
unzip fastspeech2_conformer_baker_ckpt_0.5.zip
unzip fastspeech2_nosil_ljspeech_ckpt_0.5.zip
cd ../
```
**prepare source data**
```bash
mkdir source
cd source
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/SSB03540307.wav
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/SSB03540428.wav
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/LJ050-0278.wav
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/p243_313.wav
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/p299_096.wav
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/this_was_not_the_show_for_me.wav
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/source/README.md
cd ../
```
You can check the text of downloaded wavs in `source/README.md`.
### Cross Language Voice Cloning
```bash
./run.sh --stage 3 --stop-stage 3 --gpus 0
```
`stage 3` of `run.sh` calls `local/synthesize_e2e.sh`.

You can modify  `--wav_path`、`--old_str` and `--new_str` yourself, `--old_str` should be the text corresponding to the audio of  `--wav_path`, `--new_str` should be designed according to `--task_name`, `--source_lang` and `--target_lang` should be different in this example.
## Pretrained Model
Pretrained ErnieSAT model:
- [erniesat_aishell3_vctk_ckpt_1.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/ernie_sat/erniesat_aishell3_vctk_ckpt_1.2.0.zip)

Model | Step | eval/text_mlm_loss | eval/mlm_loss | eval/loss
:-------------:| :------------:| :-----: | :-----:| :-----:
default| 8(gpu) x 489000|0.000001|52.477642 |52.477642
