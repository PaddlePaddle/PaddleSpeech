# VITS with AISHELL-3
This example contains code used to train a [VITS](https://arxiv.org/abs/2106.06103) model with [AISHELL-3](http://www.aishelltech.com/aishell_3). The trained model can be used in Voice Cloning Task, We refer to the model structure of  [Transfer Learning from Speaker Veriﬁcation to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf). The general steps are as follows:
1. Speaker Encoder: We use Speaker Verification to train a speaker encoder. Datasets used in this task are different from those used in `VITS` because the transcriptions are not needed, we use more datasets, refer to  [ge2e](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/ge2e).
2. Synthesizer and Vocoder: We use the trained speaker encoder to generate speaker embedding for each sentence in AISHELL-3. This embedding is an extra input of `VITS` which will be concated with encoder outputs. The vocoder is part of `VITS` due to its special structure.

## Dataset
### Download and Extract
Download AISHELL-3 from it's [Official Website](http://www.aishelltech.com/aishell_3) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/data_aishell3`.

### Get MFA Result and Extract
We use [MFA2.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get phonemes for VITS, the durations of MFA are not needed here.
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
│   ├── norm
│   └── raw
├── embed
│   ├── SSB0005
│   ├── SSB0009
│   ├── ...
│   └── ...
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └── raw
└── train
    ├── feats_stats.npy
    ├── norm
    └── raw
```
The `embed` contains the generated speaker embedding for each sentence in AISHELL-3, which has the same file structure with wav files and the format is  `.npy`.

The computing time of utterance embedding can be x hours.

The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains wave and linear spectrogram of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/feats_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, feats, feats_lengths, the path of linear spectrogram features, the path of raw waves, speaker, and the id of each utterance.

The preprocessing step is very similar to that one of [vits](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vits), but there is one more `ge2e/inference` step here.

### Model Training
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
The training step is very similar to that one of [vits](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vits), but we should set `--voice-cloning=True` when calling `${BIN_DIR}/train.py`.

### Synthesizing

`./local/synthesize.sh` calls `${BIN_DIR}/synthesize.py`, which can synthesize waveform from `metadata.jsonl`.

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--config CONFIG] [--ckpt CKPT]
                     [--phones_dict PHONES_DICT] [--speaker_dict SPEAKER_DICT]
                     [--voice-cloning VOICE_CLONING] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]

Synthesize with VITS

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config of VITS.
  --ckpt CKPT           Checkpoint file of VITS.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --speaker_dict SPEAKER_DICT
                        speaker id map file.
  --voice-cloning VOICE_CLONING
                        whether training voice cloning model.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --test_metadata TEST_METADATA
                        test metadata.
  --output_dir OUTPUT_DIR
                        output dir.
```
The synthesizing step is very similar to that one of [vits](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/vits), but we should set `--voice-cloning=True` when calling `${BIN_DIR}/../synthesize.py`.

### Voice Cloning
Assume there are some  reference audios in `./ref_audio`
```text
ref_audio
├── 001238.wav
├── LJ015-0254.wav
└── audio_self_test.mp3
```
`./local/voice_cloning.sh` calls `${BIN_DIR}/voice_cloning.py`

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_cloning.sh ${conf_path} ${train_output_path} ${ckpt_name} ${ge2e_params_path} ${add_blank} ${ref_audio_dir}
```

If you want to convert a speaker audio file to refered speaker, run:

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_cloning.sh ${conf_path} ${train_output_path} ${ckpt_name} ${ge2e_params_path} ${add_blank} ${ref_audio_dir} ${src_audio_path}
```

<!-- TODO display these after we trained the model -->
<!-- 
## Pretrained Model

The pretrained model can be downloaded here:

- [vits_vc_aishell3_ckpt_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/vits/vits_vc_aishell3_ckpt_1.1.0.zip) (add_blank=true)

VITS checkpoint contains files listed below.
(There is no need for `speaker_id_map.txt` here )

```text
vits_vc_aishell3_ckpt_1.1.0
├── default.yaml              # default config used to train vitx
├── phone_id_map.txt          # phone vocabulary file when training vits
└── snapshot_iter_333000.pdz  # model parameters and optimizer states
```

ps: This ckpt is not good enough, a better result is training
-->
