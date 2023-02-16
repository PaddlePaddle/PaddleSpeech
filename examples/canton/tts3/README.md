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
Pretrained FastSpeech2 model with no silence in the edge of audios:
- [fastspeech2_canton_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_canton_ckpt_1.4.0.zip)

FastSpeech2 checkpoint contains files listed below.

```text
fastspeech2_canton_ckpt_1.4.0
├── default.yaml            # default config used to train fastspeech2
├── energy_stats.npy        # statistics used to normalize energy when training fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── pitch_stats.npy         # statistics used to normalize pitch when training fastspeech2
├── snapshot_iter_140000.pdz # model parameters and optimizer states
├── speaker_id_map.txt      # speaker id map file when training a multi-speaker fastspeech2
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```
You can use the following scripts to synthesize for `${BIN_DIR}/../sentences_canton.txt` using pretrained fastspeech2 and parallel wavegan models.
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=fastspeech2_aishell3 \
  --am_config=fastspeech2_canton_ckpt_1.4.0/default.yaml \
  --am_ckpt=fastspeech2_canton_ckpt_1.4.0/snapshot_iter_140000.pdz \
  --am_stat=fastspeech2_canton_ckpt_1.4.0/speech_stats.npy \
  --voc=pwgan_aishell3 \
  --voc_config=pwg_aishell3_ckpt_0.5/default.yaml \
  --voc_ckpt=pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
  --voc_stat=pwg_aishell3_ckpt_0.5/feats_stats.npy \
  --lang=canton \
  --text=${BIN_DIR}/../sentences_canton.txt \
  --output_dir=exp/default/test_e2e \
  --phones_dict=fastspeech2_canton_ckpt_1.4.0/phone_id_map.txt \
  --speaker_dict=fastspeech2_canton_ckpt_1.4.0/speaker_id_map.txt \
  --spk_id=0 \
  --inference_dir=exp/default/inference
```
