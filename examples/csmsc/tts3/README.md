([简体中文](./README_cn.md)|English)
# FastSpeech2 with CSMSC
This example contains code used to train a [Fastspeech2](https://arxiv.org/abs/2006.04558) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).

## Dataset
### Download and Extract
Download CSMSC from it's [Official Website](https://test.data-baker.com/data/index/source).

### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2.
You can download from here [baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) of our repo.

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

### Synthesizing
We use [parallel wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc1) as the neural vocoder.
Download pretrained parallel wavegan model from [pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip) and unzip it.
```bash
unzip pwg_baker_ckpt_0.4.zip
```
Parallel WaveGAN checkpoint contains files listed below.
```text
pwg_baker_ckpt_0.4
├── pwg_default.yaml               # default config used to train parallel wavegan
├── pwg_snapshot_iter_400000.pdz   # model parameters of parallel wavegan
└── pwg_stats.npy                  # statistics used to normalize spectrogram when training parallel wavegan
```
`./local/synthesize.sh` calls `${BIN_DIR}/../synthesize.py`, which can synthesize waveform from `metadata.jsonl`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h]
                     [--am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}]
                     [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                     [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                     [--tones_dict TONES_DICT] [--speaker_dict SPEAKER_DICT]
                     [--voice-cloning VOICE_CLONING]
                     [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}]
                     [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                     [--voc_stat VOC_STAT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}
                        Choose acoustic model type of tts task.
  --am_config AM_CONFIG
                        Config of acoustic model. Use deault config when it is
                        None.
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
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc. Use deault config when it is None.
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
                         [--am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--tones_dict TONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {speedyspeech_csmsc,fastspeech2_csmsc,fastspeech2_ljspeech,fastspeech2_aishell3,fastspeech2_vctk}
                        Choose acoustic model type of tts task.
  --am_config AM_CONFIG
                        Config of acoustic model. Use deault config when it is
                        None.
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
  --voc {pwgan_csmsc,pwgan_ljspeech,pwgan_aishell3,pwgan_vctk,mb_melgan_csmsc}
                        Choose vocoder type of tts task.
  --voc_config VOC_CONFIG
                        Config of voc. Use deault config when it is None.
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
2. `--am_config`, `--am_checkpoint`, `--am_stat` and `--phones_dict` are arguments for acoustic model, which correspond to the 4 files in the fastspeech2 pretrained model.
3. `--voc` is vocoder type with the format {model_name}_{dataset}
4. `--voc_config`, `--voc_checkpoint`, `--voc_stat` are arguments for vocoder, which correspond to the 3 files in the parallel wavegan pretrained model.
5. `--lang` is the model language, which can be `zh` or `en`.
6. `--test_metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
7. `--text` is the text file, which contains sentences to synthesize.
8. `--output_dir` is the directory to save synthesized audio files.
9. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

### Inferencing
After synthesizing, we will get static models of fastspeech2 and pwgan in `${train_output_path}/inference`.
`./local/inference.sh` calls `${BIN_DIR}/inference.py`, which provides a paddle static model inference example for fastspeech2 + pwgan synthesize.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path}
```

## Pretrained Model
Pretrained FastSpeech2 model with no silence in the edge of audios:
- [fastspeech2_nosil_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip)
- [fastspeech2_conformer_baker_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_conformer_baker_ckpt_0.5.zip)
- [fastspeech2_cnndecoder_csmsc_ckpt_1.0.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_ckpt_1.0.0.zip)

The static model can be downloaded here:
- [fastspeech2_nosil_baker_static_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_static_0.4.zip)
- [fastspeech2_csmsc_static_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_static_0.2.0.zip)
- [fastspeech2_cnndecoder_csmsc_static_1.0.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_static_1.0.0.zip)
- [fastspeech2_cnndecoder_csmsc_streaming_static_1.0.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_streaming_static_1.0.0.zip)

The ONNX model can be downloaded here:
- [fastspeech2_csmsc_onnx_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_onnx_0.2.0.zip)
- [fastspeech2_cnndecoder_csmsc_onnx_1.0.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_onnx_1.0.0.zip)
- [fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip)

Model | Step | eval/loss | eval/l1_loss | eval/duration_loss | eval/pitch_loss| eval/energy_loss 
:-------------:| :------------:| :-----: | :-----: | :--------: |:--------:|:---------:
default| 2(gpu) x 76000|1.0991|0.59132|0.035815|0.31915|0.15287|
conformer| 2(gpu) x 76000|1.0675|0.56103|0.035869|0.31553|0.15509|
cnndecoder| 1(gpu) x 153000|1.1153|0.61475|0.03380|0.30414|0.14707|

FastSpeech2 checkpoint contains files listed below.
```text
fastspeech2_nosil_baker_ckpt_0.4
├── default.yaml            # default config used to train fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── snapshot_iter_76000.pdz # model parameters and optimizer states
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```
You can use the following scripts to synthesize for `${BIN_DIR}/../sentences.txt` using pretrained fastspeech2 and parallel wavegan models.

If you want to use fastspeech2_conformer, you must delete this line `--inference_dir=exp/default/inference \` to skip the step of dygraph to static graph, cause we haven't tested dygraph to static graph for fastspeech2_conformer till now.
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=fastspeech2_csmsc \
  --am_config=fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
  --am_ckpt=fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
  --am_stat=fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy  \
  --voc=pwgan_csmsc \
  --voc_config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --voc_ckpt=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --voc_stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --lang=zh \
  --text=${BIN_DIR}/../sentences.txt \
  --output_dir=exp/default/test_e2e \
  --inference_dir=exp/default/inference \
  --phones_dict=fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt
```
