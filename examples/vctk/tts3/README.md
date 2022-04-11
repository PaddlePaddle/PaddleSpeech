# FastSpeech2 with the VCTK
This example contains code used to train a [Fastspeech2](https://arxiv.org/abs/2006.04558) model with [VCTK](https://datashare.ed.ac.uk/handle/10283/3443).

## Dataset
### Download and Extract the dataset
Download VCTK-0.92 from the [official website](https://datashare.ed.ac.uk/handle/10283/3443).

### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2.
You can download from here [vctk_alignment.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/VCTK-Corpus-0.92/vctk_alignment.tar.gz), or train your MFA model reference to [mfa example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/other/mfa) of our repo.
ps: we remove three speakers in VCTK-0.92 (see [reorganize_vctk.py](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/other/mfa/local/reorganize_vctk.py)):
1. `p315`, because of no text for it.
2. `p280` and `p362`, because no *_mic2.flac (which is better than *_mic1.flac) for  them.

## Get Started
Assume the path to the dataset is `~/datasets/VCTK-Corpus-0.92`.
Assume the path to the MFA result of VCTK is `./vctk_alignment`.
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

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, speech_lengths, durations, the path of speech features, the path of pitch features, the path of energy features, speaker, and id of each utterance.

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
4. `--phones-dict` is the path of the phone vocabulary file.

### Synthesizing
We use [parallel wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/voc1) as the neural vocoder.

Download pretrained parallel wavegan model from [pwg_vctk_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_vctk_ckpt_0.1.1.zip) and unzip it.
```bash
unzip pwg_vctk_ckpt_0.1.1.zip
```
Parallel WaveGAN checkpoint contains files listed below.
```text
pwg_vctk_ckpt_0.1.1
├── default.yaml                   # default config used to train parallel wavegan
├── snapshot_iter_1500000.pdz      # generator parameters of parallel wavegan
└── feats_stats.npy                # statistics used to normalize spectrogram when training parallel wavegan
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
2. `--am_config`, `--am_checkpoint`, `--am_stat`, `--phones_dict` `--speaker_dict` are arguments for acoustic model, which correspond to the 5 files in the fastspeech2 pretrained model.
3. `--voc` is vocoder type with the format {model_name}_{dataset}
4. `--voc_config`, `--voc_checkpoint`, `--voc_stat` are arguments for vocoder, which correspond to the 3 files in the parallel wavegan pretrained model.
5. `--lang` is the model language, which can be `zh` or `en`.
6. `--test_metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
7. `--text` is the text file, which contains sentences to synthesize.
8. `--output_dir` is the directory to save synthesized audio files.
9. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

## Pretrained Model
Pretrained FastSpeech2 model with no silence in the edge of audios:
- [fastspeech2_nosil_vctk_ckpt_0.5.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_vctk_ckpt_0.5.zip)

FastSpeech2 checkpoint contains files listed below.
```text
fastspeech2_nosil_vctk_ckpt_0.5
├── default.yaml            # default config used to train fastspeech2
├── phone_id_map.txt        # phone vocabulary file when training fastspeech2
├── snapshot_iter_66200.pdz # model parameters and optimizer states
├── speaker_id_map.txt      # speaker id map file when training a multi-speaker fastspeech2
└── speech_stats.npy        # statistics used to normalize spectrogram when training fastspeech2
```
You can use the following scripts to synthesize for `${BIN_DIR}/../sentences.txt` using pretrained fastspeech2 and parallel wavegan models.
```bash
source path.sh

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
  --am=fastspeech2_vctk \
  --am_config=fastspeech2_nosil_vctk_ckpt_0.5/default.yaml \
  --am_ckpt=fastspeech2_nosil_vctk_ckpt_0.5/snapshot_iter_66200.pdz \
  --am_stat=fastspeech2_nosil_vctk_ckpt_0.5/speech_stats.npy \
  --voc=pwgan_vctk \
  --voc_config=pwg_vctk_ckpt_0.1.1/default.yaml  \
  --voc_ckpt=pwg_vctk_ckpt_0.1.1/snapshot_iter_1500000.pdz \
  --voc_stat=pwg_vctk_ckpt_0.1.1/feats_stats.npy \
  --lang=en \
  --text=${BIN_DIR}/../sentences_en.txt \
  --output_dir=exp/default/test_e2e \
  --phones_dict=fastspeech2_nosil_vctk_ckpt_0.5/phone_id_map.txt \
  --speaker_dict=fastspeech2_nosil_vctk_ckpt_0.5/speaker_id_map.txt \
  --spk_id=0 \
  --inference_dir=exp/default/inference
```
