# HiFiGAN with CSMSC
This example contains code used to train a [HiFiGAN](https://arxiv.org/abs/2010.05646) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).
## Dataset
### Download and Extract
Download CSMSC from it's [official website](https://test.data-baker.com/data/index/TNtts/) and extract it to `~/datasets`. Then the dataset is in the directory `~/datasets/BZNSYP`.

The structure of the folder is listed below.

```text
└─ Wave
    └─ .wav files (audio speech)
└─ PhoneLabeling
    └─ .interval files (alignment between phoneme and duration)
└─ ProsodyLabeling
   └─ 000001-010000.txt (text with prosodic by pinyin)
```

### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) results to cut silence at the edge of audio.
You can download from here [baker_alignment_tone.tar.gz](https://paddlespeech.cdn.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) of our repo.

## Get Started
Assume the path to the dataset is `~/datasets/BZNSYP`.
Assume the path to the MFA result of CSMSC is `./baker_alignment_tone`.
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
├── test
│   ├── norm
│   └── raw
└── train
    ├── norm
    ├── raw
    └── feats_stats.npy
```
The dataset is split into 3 parts, namely `train`, `dev`, and `test`, each of which contains a `norm` and `raw` subfolder. The `raw` folder contains the log magnitude of the mel spectrogram of each utterance, while the norm folder contains the normalized spectrogram. The statistics used to normalize the spectrogram are computed from the training set, which is located in `dump/train/feats_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains id and paths to the spectrogram of each utterance.

### Model Training
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path}
```
`./local/train.sh` calls `${BIN_DIR}/train.py`.
Here's the complete help message.

```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--ngpu NGPU]

Train a HiFiGAN model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       HiFiGAN config file.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.
```

1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are saved in `checkpoints/` inside this directory.
4. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

### Synthesizing
We use [HiFiGAN](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc5) as the neural vocoder.

Download pretrained HiFiGAN model from [hifigan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip) and unzip it.
```bash
unzip hifigan_csmsc_ckpt_0.1.1.zip
```
HiFiGAN checkpoint contains files listed below.
```text
hifigan_csmsc_ckpt_0.1.1
├── default.yaml                    # default config used to train HiFiGAN
├── feats_stats.npy                 # statistics used to normalize spectrogram when training HiFiGAN
└── snapshot_iter_2500000.pdz       # generator parameters of HiFiGAN
```
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--generator-type GENERATOR_TYPE] [--config CONFIG]
                     [--checkpoint CHECKPOINT] [--test-metadata TEST_METADATA]
                     [--output-dir OUTPUT_DIR] [--ngpu NGPU]

Synthesize with GANVocoder.

optional arguments:
  -h, --help            show this help message and exit
  --generator-type GENERATOR_TYPE
                        type of GANVocoder, should in {pwgan, mb_melgan,
                        style_melgan, } now
  --config CONFIG       GANVocoder config file.
  --checkpoint CHECKPOINT
                        snapshot to load.
  --test-metadata TEST_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.
```

1. `--config` config file. You should use the same config with which the model is trained.
2. `--checkpoint` is the checkpoint to load. Pick one of the checkpoints from `checkpoints` inside the training output directory.
3. `--test-metadata` is the metadata of the test dataset. Use the `metadata.jsonl` in the `dev/norm` subfolder from the processed directory.
4. `--output-dir` is the directory to save the synthesized audio files.
5. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

We use [Fastspeech2](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts3) as the acoustic model.
Download pretrained fastspeech2_nosil model from [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip)and unzip it.
```bash
unzip fastspeech2_nosil_baker_ckpt_0.4.zip
```
Fastspeech2 checkpoint contains files listed below.
```text
fastspeech2_nosil_baker_ckpt_0.4
├── default.yaml            # default config used to train fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── snapshot_iter_76000.pdz # model parameters and optimizer states
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```

`./local/synthesize_e2e.sh` calls `${BIN_DIR}/../../synthesize_e2e.py`, which can synthesize waveform from text file.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h]
                         [--am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}]
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
  --am {speedyspeech_csmsc,speedyspeech_aishell3,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk,tacotron2_csmsc,tacotron2_ljspeech}
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
  --lang LANG           Choose model language. zh or en
  --inference_dir INFERENCE_DIR
                        dir to save inference models
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line.
  --output_dir OUTPUT_DIR
                        output dir.
```
1. `--am` is acoustic model type with the format {model_name}_{dataset}
2. `--am_config`, `--am_ckpt`, `--am_stat` and `--phones_dict` are arguments for acoustic model, which correspond to the 4 files in the fastspeech2 pretrained model.
3. `--voc` is vocoder type with the format {model_name}_{dataset}
4. `--voc_config`, `--voc_ckpt`, `--voc_stat` are arguments for vocoder, which correspond to the 3 files in the parallel wavegan pretrained model.
5. `--lang` is the model language, which can be `zh` or `en`.
6. `--test_metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
7. `--text` is the text file, which contains sentences to synthesize.
8. `--output_dir` is the directory to save synthesized audio files.
9. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

## Pretrained Models
The pretrained model can be downloaded here:
- [hifigan_csmsc_ckpt_0.1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip)
- [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip)

The static model can be downloaded here:
- [hifigan_csmsc_static_0.1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_static_0.1.1.zip)
- [fastspeech2_nosil_baker_static_0.4.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_static_0.4.zip)

The PIR static model can be downloaded here:
- [hifigan_csmsc_static_pir_0.1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_static_pir_0.1.1.zip) (Run PIR model need to set FLAGS_enable_pir_api=1, and PIR model only worked with paddlepaddle>=3.0.0b2)

The ONNX model can be downloaded here:
- [hifigan_csmsc_onnx_0.2.0.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_onnx_0.2.0.zip)
- [fastspeech2_csmsc_onnx_0.2.0.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_onnx_0.2.0.zip)

The Paddle-Lite model can be downloaded here:
- [hifigan_csmsc_pdlite_1.3.0.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_pdlite_1.3.0.zip)
- [fastspeech2_csmsc_pdlite_1.3.0.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_pdlite_1.3.0.zip)

Model | Step | eval/generator_loss | eval/mel_loss| eval/feature_matching_loss
:-------------:| :------------:| :-----: | :-----: | :--------:
default| 1(gpu) x 2500000|24.927|0.1262|7.554

HiFiGAN checkpoint contains files listed below.

```text
hifigan_csmsc_ckpt_0.1.1
├── default.yaml                  # default config used to train hifigan
├── feats_stats.npy               # statistics used to normalize spectrogram when training hifigan
└── snapshot_iter_2500000.pdz     # generator parameters of hifigan
```

FastSpeech2 checkpoint contains files listed below.

```text
fastspeech2_nosil_baker_ckpt_0.4
├── default.yaml            # default config used to train fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── snapshot_iter_76000.pdz # model parameters and optimizer states
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```

## Acknowledgement
We adapted some code from https://github.com/kan-bayashi/ParallelWaveGAN.
