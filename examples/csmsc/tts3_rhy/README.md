# This example mainly follows the FastSpeech2 with CSMSC
This example contains code used to train a rhythm version of [Fastspeech2](https://arxiv.org/abs/2006.04558) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).

## Dataset
### Download and Extract
Download CSMSC from it's [Official Website](https://test.data-baker.com/data/index/TNtts/) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/BZNSYP`.

### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2.
You can directly download the rhythm version of MFA result from here [baker_alignment_tone.zip](https://paddlespeech.bj.bcebos.com/Rhy_e2e/baker_alignment_tone.zip), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) of our repo.
Remember in our repo, you should add `--rhy-with-duration` flag to obtain the rhythm information.

## Get Started
Assume the path to the dataset is `~/datasets/BZNSYP`.
Assume the path to the MFA result of CSMSC is `./baker_alignment_tone`.
Run the command below to
1. **source path**.
2. preprocess the dataset.
3. train the model.
4. synthesize wavs.
    - synthesize waveform from `metadata.jsonl`.
    - synthesize waveform from a text file.
5. inference using the static model.
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
    └── speech_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains speech、pitch and energy features of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, speech_lengths, durations, the path of speech features, the path of pitch features, the path of energy features, speaker, and the id of each utterance.

# For more details, You can refer to [FastSpeech2 with CSMSC](../tts3)

## Pretrained Model
Pretrained FastSpeech2 model for end-to-end rhythm version:
- [fastspeech2_rhy_csmsc_ckpt_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_rhy_csmsc_ckpt_1.3.0.zip)

This FastSpeech2 checkpoint contains files listed below.
```text
fastspeech2_rhy_csmsc_ckpt_1.3.0
├── default.yaml             # default config used to train fastspeech2
├── phone_id_map.txt         # phone vocabulary file when training fastspeech2
├── snapshot_iter_153000.pdz # model parameters and optimizer states
├── durations.txt            # the intermediate output of preprocess.sh
├── energy_stats.npy
├── pitch_stats.npy
└── speech_stats.npy         # statistics used to normalize spectrogram when training fastspeech2
```
