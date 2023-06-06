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
  --config CONFIG       diffsinger config file.
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
       {diffsinger_opencpop} Choose acoustic model type of svs task.
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
        {pwgan_opencpop, hifigan_opencpop} Choose vocoder type of svs task.
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
  --speech-stretchs     SPEECH_STRETCHS
                        The min and max values of the mel spectrum, using on diffusion of diffsinger.
```

`./local/synthesize_e2e.sh` calls `${BIN_DIR}/../synthesize_e2e.py`, which can synthesize waveform from text file. 
`local/pinyin_to_phone.txt` comes from the readme of the opencpop dataset, indicating the mapping from pinyin to phonemes in opencpop.

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h]
                         [--am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]
                         [--pinyin_phone PINYIN_PHONE]
                         [--speech_stretchs SPEECH_STRETCHS]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}
                        Choose acoustic model type of tts task.
       {diffsinger_opencpop} Choose acoustic model type of svs task.
  --am_config AM_CONFIG
                        Config of acoustic model.
  --am_ckpt AM_CKPT     Checkpoint file of acoustic model.
  --am_stat AM_STAT     mean and standard deviation used to normalize
                        spectrogram when training acoustic model.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --speaker_dict SPEAKER_DICT
                        speaker id map file.
  --spk_id SPK_ID       spk id for multi speaker acoustic model
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}
                        Choose vocoder type of tts task.
        {pwgan_opencpop, hifigan_opencpop} Choose vocoder type of svs task.
  --voc_config VOC_CONFIG
                        Config of voc.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --lang LANG           {zh, en, mix, canton} Choose language type of tts task.
                        {sing} Choose language type of svs task.
  --inference_dir INFERENCE_DIR
                        dir to save inference models
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           text to synthesize file, a 'utt_id sentence' pair per line for tts task.
                        A '{ utt_id input_type (is word) text notes note_durs}' or '{utt_id input_type (is phoneme) phones notes note_durs is_slurs}' pair per line for svs task.
  --output_dir OUTPUT_DIR
                        output dir.
  --pinyin_phone PINYIN_PHONE
                        pinyin to phone map file, using on sing_frontend.
  --speech_stretchs SPEECH_STRETCHS
                        The min and max values of the mel spectrum, using on diffusion of diffsinger.
```
1. `--am` is acoustic model type with the format {model_name}_{dataset}
2. `--am_config`, `--am_ckpt`, `--am_stat` and `--phones_dict` are arguments for acoustic model, which correspond to the 4 files in the diffsinger pretrained model.
3. `--voc` is vocoder type with the format {model_name}_{dataset}
4. `--voc_config`, `--voc_ckpt`, `--voc_stat` are arguments for vocoder, which correspond to the 3 files in the parallel wavegan pretrained model.
5. `--lang` is language. `zh`, `en`, `mix` and `canton` for tts task. `sing` for tts task.
6. `--test_metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
7. `--text` is the text file, which contains sentences to synthesize.
8. `--output_dir` is the directory to save synthesized audio files.
9. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.
10. `--inference_dir` is the directory to save static models. If this line is not added, it will not be generated and saved as a static model.
11. `--pinyin_phone` pinyin to phone map file, using on sing_frontend.
12. `--speech_stretchs` The min and max values of the mel spectrum, using on diffusion of diffsinger.

Note: At present, the diffsinger model does not support dynamic to static, so do not add `--inference_dir`.


## Pretrained Model
Pretrained DiffSinger model:
- [diffsinger_opencpop_ckpt_1.4.0.zip](https://paddlespeech.bj.bcebos.com/t2s/svs/opencpop/diffsinger_opencpop_ckpt_1.4.0.zip)

DiffSinger checkpoint contains files listed below.
```text
diffsinger_opencpop_ckpt_1.4.0.zip
├── default.yaml             # default config used to train diffsinger
├── energy_stats.npy         # statistics used to normalize energy when training diffsinger if norm is needed
├── phone_id_map.txt         # phone vocabulary file when training diffsinger
├── pinyin_to_phone.txt      # pinyin-to-phoneme mapping file when training diffsinger
├── pitch_stats.npy          # statistics used to normalize pitch when training diffsinger if norm is needed 
├── snapshot_iter_160000.pdz # model parameters of diffsinger
├── speech_stats.npy         # statistics used to normalize mel when training diffsinger if norm is needed
└── speech_stretchs.npy      # min and max values to use for mel spectral stretching before training diffusion

```

You can use the following scripts to synthesize for `${BIN_DIR}/../sentences_sing.txt` using pretrained diffsinger and parallel wavegan models.

```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=diffsinger_opencpop \
  --am_config=diffsinger_opencpop_ckpt_1.4.0/default.yaml \
  --am_ckpt=diffsinger_opencpop_ckpt_1.4.0/snapshot_iter_160000.pdz \
  --am_stat=diffsinger_opencpop_ckpt_1.4.0/speech_stats.npy  \
  --voc=pwgan_opencpop \
  --voc_config=pwgan_opencpop_ckpt_1.4.0/default.yaml \
  --voc_ckpt=pwgan_opencpop_ckpt_1.4.0/snapshot_iter_100000.pdz \
  --voc_stat=pwgan_opencpop_ckpt_1.4.0/feats_stats.npy \
  --lang=sing \
  --text=${BIN_DIR}/../../assets/sentences_sing.txt \
  --output_dir=exp/default/test_e2e \
  --phones_dict=diffsinger_opencpop_ckpt_1.4.0/phone_id_map.txt \
  --pinyin_phone=diffsinger_opencpop_ckpt_1.4.0/pinyin_to_phone.txt \
  --speech_stretchs=diffsinger_opencpop_ckpt_1.4.0/speech_stretchs.npy
  
```
