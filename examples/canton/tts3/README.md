# FastSpeech2 with Cantonese language

## Dataset
### Download and Extract
If you don't have the Cantonese datasets mentioned above, please download and unzip  [Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence](https://magichub.com/datasets/guangzhou-cantonese-scripted-speech-corpus-daily-use-sentence/) and [Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle](https://magichub.com/datasets/guangzhou-cantonese-scripted-speech-corpus-in-the-vehicle/) under `~/datasets/`.

To obtain better performance, please combine these two datasets together as follows:

```bash
mkdir -p ~/datasets/canton_all/WAV
cp -r ~/datasets/Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence/WAV/* ~/datasets/canton_all/WAV
cp -r ~/datasets/Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle/WAV/* ~/datasets/canton_all/WAV
```

After that, it should be look like:
```
~/datasets/canton_all
│   └── WAV
│       └──G0001
│       └──G0002
│       ...
│       └──G0071
│       └──G0072
```


### Get MFA Result and Extract
We use [MFA1.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for canton_fastspeech2.
You can train your MFA model reference to [canton_mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) (use MFA1.x now) of our repo.
We here provide the MFA results of these two datasets. [canton_alignment.zip](https://paddlespeech.bj.bcebos.com/MFA/Canton/canton_alignment.zip)

## Get Started
Assume the path to the Cantonese MFA result of the two datsets mentioned above is `./canton_alignment`.
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
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains speech、pitch and energy features of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, speech_lengths, durations, the path of speech features, the path of pitch features, a path of energy features, speaker, and id of each utterance.

### Training details can refer to the script of [examples/aishell3/tts3](../../aishell3/tts3).

## Pretrained Model
