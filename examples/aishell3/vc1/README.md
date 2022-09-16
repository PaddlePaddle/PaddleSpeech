# FastSpeech2 + AISHELL-3 Voice Cloning
This example contains code used to train a [FastSpeech2](https://arxiv.org/abs/2006.04558) model with [AISHELL-3](http://www.aishelltech.com/aishell_3). The trained model can be used in Voice Cloning Task, We refer to the model structure of  [Transfer Learning from Speaker Veriﬁcation to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf). The general steps are as follows:
1. Speaker Encoder: We use Speaker Verification to train a speaker encoder. Datasets used in this task are different from those used in `FastSpeech2` because the transcriptions are not needed, we use more datasets, refer to  [ge2e](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/ge2e).
2. Synthesizer: We use the trained speaker encoder to generate speaker embedding for each sentence in AISHELL-3. This embedding is an extra input of  `FastSpeech2` which will be concated with encoder outputs.
3. Vocoder: We use [Parallel Wave GAN](http://arxiv.org/abs/1910.11480) as the neural Vocoder, refer to [voc1](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc1).

## Dataset
### Download and Extract
Download AISHELL-3 from it's [Official Website](http://www.aishelltech.com/aishell_3) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/data_aishell3`.

### Get MFA Result and Extract
We use [MFA2.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for aishell3_fastspeech2.
You can download from here [aishell3_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) (use MFA1.x now) of our repo.

## Pretrained GE2E Model
We use pretrained GE2E model to generate speaker embedding for each sentence.

Download pretrained GE2E model from here [ge2e_ckpt_0.3.zip](https://bj.bcebos.com/paddlespeech/Parakeet/released_models/ge2e/ge2e_ckpt_0.3.zip), and `unzip` it.

## Get Started
Assume the path to the dataset is `~/datasets/data_aishell3`.
Assume the path to the MFA result of AISHELL-3 is `./aishell3_alignment_tone`.
Assume the path to the pretrained ge2e model is `./ge2e_ckpt_0.3`.

Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize waveform from `metadata.jsonl`.
5. start a voice cloning inference.
```bash
./run.sh
```
You can choose a range of stages you want to run, or set `stage` equal to `stop-stage` to use only one stage, for example, running the following command will only preprocess the dataset.
```bash
./run.sh --stage 0 --stop-stage 0
```
### Data Preprocessing
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${conf_path} ${ge2e_ckpt_path}
```
When it is done. A `dump` folder is created in the current directory. The structure of the dump folder is listed below.
```text
dump
├── dev
│   ├── norm
│   └── raw
├── embed
│   ├── SSB0005
│   ├── SSB0009
│   ├── ...
│   └── ...
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └──  raw
└── train
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```
The `embed` contains the generated speaker embedding for each sentence in AISHELL-3, which has the same file structure with wav files and the format is  `.npy`.

The computing time of utterance embedding can be x hours.

The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains speech、pitch and energy features of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, speech_lengths, durations, the path of speech features, the path of pitch features, the path of energy features, speaker, and id of each utterance.

The preprocessing step is very similar to that one of [tts3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3), but there is one more `ge2e/inference` step here.

### Model Training
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
The training step is very similar to that one of [tts3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3), but we should set `--voice-cloning=True` when calling `${BIN_DIR}/train.py`.

### Synthesizing
We use [parallel wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc1) as the neural vocoder.
Download pretrained parallel wavegan model from [pwg_aishell3_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip) and unzip it.
```bash
unzip pwg_aishell3_ckpt_0.5.zip
```
Parallel WaveGAN checkpoint contains files listed below.
```text
pwg_aishell3_ckpt_0.5
├── default.yaml                   # default config used to train parallel wavegan
├── feats_stats.npy                # statistics used to normalize spectrogram when training parallel wavegan
└── snapshot_iter_1000000.pdz      # generator parameters of parallel wavegan
```
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
The synthesizing step is very similar to that one of [tts3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3), but we should set `--voice-cloning=True` when calling `${BIN_DIR}/../synthesize.py`.

### Voice Cloning
Assume there are some reference audios in `./ref_audio`
```text
ref_audio
├── 001238.wav
├── LJ015-0254.wav
└── audio_self_test.mp3
```
`./local/voice_cloning.sh` calls `${BIN_DIR}/../voice_cloning.py`

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_cloning.sh ${conf_path} ${train_output_path} ${ckpt_name} ${ge2e_params_path} ${ref_audio_dir}
```
## Pretrained Model
- [fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_vc1_ckpt_0.5.zip)

Model | Step | eval/loss | eval/l1_loss | eval/duration_loss | eval/pitch_loss| eval/energy_loss 
:-------------:| :------------:| :-----: | :-----: | :--------: |:--------:|:---------:
default|2(gpu) x 96400|0.99699|0.62013|0.053057|0.11954| 0.20426|

FastSpeech2 checkpoint contains files listed below.
(There is no need for `speaker_id_map.txt` here )

```text
fastspeech2_nosil_aishell3_ckpt_vc1_0.5
├── default.yaml            # default config used to train fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── snapshot_iter_96400.pdz # model parameters and optimizer states
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```
