
# Mixed Chinese and English TTS with CSMSC, LJSpeech-1.1, AISHELL-3 and VCTK datasets

This example contains code used to train a [Fastspeech2](https://arxiv.org/abs/2006.04558) model with [CSMSC](https://www.data-baker.com/open_source.html), [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/), [AISHELL3](http://www.aishelltech.com/aishell_3) and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) datasets.


## Dataset
### Download and Extract
Download all datasets and extract it to `./data`:
- The CSMSC dataset is in the directory `./data/BZNSYP`
- The Ljspeech dataset is in the directory `./data/LJSpeech-1.1`
- The aishell3 dataset is in the directory `./data/data_aishell3`
- The vctk dataset is in the directory `./data/VCTK-Corpus-0.92`
 
### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for the fastspeech2 training.
You can download from here:
- [baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz)
- [ljspeech_alignment.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/LJSpeech-1.1/ljspeech_alignment.tar.gz)
- [aishell3_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz) 
- [vctk_alignment.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/VCTK-Corpus-0.92/vctk_alignment.tar.gz)

Or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) (use MFA1.x now) of our repo.

## Get Started
Assume the paths to the datasets are:
- `./data/BZNSYP`
- `./data/LJSpeech-1.1`
- `./data/data_aishell3` 
- `./data/VCTK-Corpus-0.92`

Assume the path to the MFA results of the datasets are:
- `./data/mfa/baker_alignment_tone`
- `./data/mfa/ljspeech_alignment`
- `./data/mfa/aishell3_alignment_tone`
- `./data/mfa/vctk_alignment`

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
./local/preprocess.sh ${conf_path} ${datasets_root_dir} ${mfa_root_dir}
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
The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains speech, pitch and energy features of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/*_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, speech_lengths, durations, the path of speech features, the path of pitch features, a path of energy features, speaker, and id of each utterance.


### Model Training
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
Here's the complete help message.
```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU] [--phones-dict PHONES_DICT]
                [--speaker-dict SPEAKER_DICT] [--voice-cloning VOICE_CLONING]

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
  --voice-cloning VOICE_CLONING
                        whether training voice cloning model.
```
1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are saved in `checkpoints/` inside this directory.
4. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.
5. `--phones-dict` is the path of the phone vocabulary file.
6. `--speaker-dict` is the path of the speaker id map file when training a multi-speaker FastSpeech2.

We have **added module speaker classifier** with reference to [Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning](https://arxiv.org/pdf/1907.04448.pdf). The main parameter configuration: `config["model"]["enable_speaker_classifier"]`, `config["model"]["hidden_sc_dim"]` and `config["updater"]["spk_loss_scale"]` in `conf/default.yaml`. The current experimental results show that this module can decouple text information and speaker information, and more experiments are still being sorted out. This module is currently not enabled by default, if you are interested, you can try it yourself.


### Synthesizing
We use [parallel wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc1) as the default neural vocoder.
Download the pretrained parallel wavegan model from [pwg_aishell3_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip) and unzip it.

When speaker is `174` (csmsc), use csmsc's vocoder is better than aishell3's, we recommend that you use [hifigan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip), please check `stage 2`  of `synthesize_e2e.sh`.

But if speaker is `175` (ljspeech), we **don't** recommend you to use ljspeech's vocoder, because ljspeech's vocoders are trained on sample rate 22.05kHz, but this acoustic model is trained on sample rate 24kHz, you can use csmsc's vocoder also, because ljspeech and csmsc are both female speakers.

For speakers in aishell3 and vctk, we recommend you use aishell3 or vctk's vocoders, because ljspeech and csmsc are both female speakers, there vocoders may not perform well for male speakers in aishell3 and vctk, you can check speaker name and spk_id in `dump/speaker_id_map.txt` and check speakers' information ( Age / Gender / Accents / region, etc ) in [this issue](https://github.com/PaddlePaddle/PaddleSpeech/issues/1620) and choose the `spk_id` you want.


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
```text
usage: synthesize.py [-h]
                     [--am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech,tacotron2_aishell3, fastspeech2_mix}]
                     [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                     [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                     [--tones_dict TONES_DICT] [--speaker_dict SPEAKER_DICT]
                     [--voice-cloning VOICE_CLONING]
                     [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,wavernn_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,style_melgan_csmsc}]
                     [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                     [--voc_stat VOC_STAT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech,tacotron2_aishell3, fastspeech2_mix}
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


```
`./local/synthesize_e2e.sh` calls `${BIN_DIR}/../synthesize_e2e.py`, which can synthesize waveform from text file.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h]
                         [--am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech, fastspeech2_mix}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--tones_dict TONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech, fastspeech2_mix}
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
  --spk_id SPK_ID       spk id for multi speaker acoustic model
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc,style_melgan_csmsc,hifigan_csmsc,hifigan_ljspeech,hifigan_aishell3,hifigan_vctk,wavernn_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc.
  --voc_ckpt VOC_CKPT   Checkpoint file of voc.
  --voc_stat VOC_STAT   mean and standard deviation used to normalize
                        spectrogram when training voc.
  --lang LANG           Choose model language. zh or en or mix
  --inference_dir INFERENCE_DIR
                        dir to save inference models
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line.
  --output_dir OUTPUT_DIR
                        output dir.
```
1. `--am` is acoustic model type with the format {model_name}_{dataset}
2. `--am_config`, `--am_ckpt`, `--am_stat`, `--phones_dict` `--speaker_dict` are arguments for acoustic model, which correspond to the 5 files in the fastspeech2 pretrained model.
3. `--voc` is vocoder type with the format {model_name}_{dataset}
4. `--voc_config`, `--voc_ckpt`, `--voc_stat` are arguments for vocoder, which correspond to the 3 files in the parallel wavegan pretrained model.
5. `--lang` is the model language, which can be `zh` or `en` or `mix`.
6. `--test_metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
7. `--text` is the text file, which contains sentences to synthesize.
8. `--output_dir` is the directory to save synthesized audio files.
9. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.


## Pretrained Model
Pretrained FastSpeech2 model with no silence in the edge of audios:
- [fastspeech2_mix_ckpt_1.2.0.zip](https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_ckpt_1.2.0.zip)

The static model can be downloaded here:
- [fastspeech2_mix_static_0.2.0.zip](https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_static_0.2.0.zip)

The ONNX model can be downloaded here:
- [fastspeech2_mix_onnx_0.2.0.zip](https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_onnx_0.2.0.zip)

FastSpeech2 checkpoint contains files listed below.

```text
fastspeech2_mix_ckpt_1.2.0
├── default.yaml            # default config used to train fastspeech2
├── energy_stats.npy        # statistics used to energy spectrogram when training fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── pitch_stats.npy         # statistics used to normalize pitch when training fastspeech2
├── snapshot_iter_99200.pdz # model parameters and optimizer states
├── speaker_id_map.txt      # speaker id map file when training a multi-speaker fastspeech2
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```


You can use the following scripts to synthesize for `${BIN_DIR}/../sentences_mix.txt` using pretrained fastspeech2 and parallel wavegan models.
`174` means baker speaker, `175` means ljspeech speaker. For other speaker information, please see `speaker_id_map.txt`.

```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=fastspeech2_mix \
  --am_config=fastspeech2_mix_ckpt_1.2.0/default.yaml \
  --am_ckpt=fastspeech2_mix_ckpt_1.2.0/snapshot_iter_99200.pdz \
  --am_stat=fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
  --phones_dict=fastspeech2_mix_ckpt_1.2.0/phone_id_map.txt \
  --speaker_dict=fastspeech2_mix_ckpt_1.2.0/speaker_id_map.txt \
  --spk_id=174 \
  --voc=pwgan_aishell3 \
  --voc_config=pwg_aishell3_ckpt_0.5/default.yaml \
  --voc_ckpt=pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
  --voc_stat=pwg_aishell3_ckpt_0.5/feats_stats.npy \
  --lang=mix \
  --text=${BIN_DIR}/../sentences_mix.txt \
  --output_dir=exp/default/test_e2e \
  --inference_dir=exp/default/inference
```
