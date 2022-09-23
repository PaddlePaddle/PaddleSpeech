# Finetune your own AM based on FastSpeech2 with multi-speakers dataset.
This example shows how to finetune your own AM based on FastSpeech2 with multi-speakers dataset. For finetuning Chinese data, we use part of csmsc's data (top 200) and Fastspeech2 pretrained model with AISHELL-3. For finetuning English data, we use part of ljspeech's data (top 200) and Fastspeech2 pretrained model with VCTK. The example is implemented according to this [discussion](https://github.com/PaddlePaddle/PaddleSpeech/discussions/1842). Thanks to the developer for the idea.

For more information on training Fastspeech2 with AISHELL-3, You can refer [examples/aishell3/tts3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3). For more information on training Fastspeech2 with VCTK, You can refer [examples/vctk/tts3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/tts3).


## Prepare 
### Download Pretrained model
Assume the path to the model is `./pretrained_models`. </br>
If you want to finetune Chinese data, you need to download Fastspeech2 pretrained model with AISHELL-3: [fastspeech2_aishell3_ckpt_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_ckpt_1.1.0.zip) for finetuning. Download HiFiGAN pretrained model with aishell3: [hifigan_aishell3_ckpt_0.2.0](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip) for synthesis.

```bash
mkdir -p pretrained_models && cd pretrained_models
# pretrained fastspeech2 model
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_ckpt_1.1.0.zip 
unzip fastspeech2_aishell3_ckpt_1.1.0.zip
# pretrained hifigan model
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip
unzip hifigan_aishell3_ckpt_0.2.0.zip
cd ../
```


If you want to finetune English data, you need to download Fastspeech2 pretrained model with VCTK: [fastspeech2_vctk_ckpt_1.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_vctk_ckpt_1.2.0.zip) for finetuning. Download HiFiGAN pretrained model with VCTK: [hifigan_vctk_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip) for synthesis.

```bash
mkdir -p pretrained_models && cd pretrained_models
# pretrained fastspeech2 model
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_vctk_ckpt_1.2.0.zip 
unzip fastspeech2_vctk_ckpt_1.2.0.zip
# pretrained hifigan model
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip
unzip hifigan_vctk_ckpt_0.2.0.zip
cd ../
```

### Download MFA tools and pretrained model
Assume the path to the MFA tool is `./tools`. Download [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz).

```bash
mkdir -p tools && cd tools
# mfa tool
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
tar xvf montreal-forced-aligner_linux.tar.gz
cp montreal-forced-aligner/lib/libpython3.6m.so.1.0 montreal-forced-aligner/lib/libpython3.6m.so
mkdir -p aligner && cd aligner
```

If you want to finetune Chinese data, you need to download pretrained MFA models with aishell3: [aishell3_model.zip](https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/aishell3_model.zip) and unzip it.

```bash
# pretrained mfa model for Chinese data
wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/aishell3_model.zip
unzip aishell3_model.zip
wget https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/simple.lexicon
cd ../../
```

If you want to finetune English data, you need to download pretrained MFA models with vctk: [vctk_model.zip](https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/vctk_model.zip) and unzip it.

```bash
# pretrained mfa model for Chinese data
wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/vctk_model.zip
unzip vctk_model.zip
wget https://paddlespeech.bj.bcebos.com/MFA/LJSpeech-1.1/cmudict-0.7b
cd ../../
```

### Prepare your data
Assume the path to the dataset is `./input` which contains a speaker folder. Speaker folder contains audio files (*.wav) and label file (labels.txt). The format of the audio file is wav. The format of the label file is: utt_id|pronunciation. </br>

If you want to finetune Chinese data, Chinese label example: 000001|ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1</br>
Here is an example of the first 200 data of csmsc.

```bash
mkdir -p input && cd input
wget https://paddlespeech.bj.bcebos.com/datasets/csmsc_mini.zip
unzip csmsc_mini.zip
cd ../
```

When "Prepare" done. The structure of the current directory is listed below.
```text
├── input
│   ├── csmsc_mini
│   │   ├── 000001.wav
│   │   ├── 000002.wav
│   │   ├── 000003.wav
│   │   ├── ...
│   │   ├── 000200.wav
│   │   ├── labels.txt
│   └── csmsc_mini.zip
├── pretrained_models
│   ├── fastspeech2_aishell3_ckpt_1.1.0
│   │   ├── default.yaml
│   │   ├── energy_stats.npy
│   │   ├── phone_id_map.txt
│   │   ├── pitch_stats.npy
│   │   ├── snapshot_iter_96400.pdz
│   │   ├── speaker_id_map.txt
│   │   └── speech_stats.npy
│   ├── fastspeech2_aishell3_ckpt_1.1.0.zip
│   ├── hifigan_aishell3_ckpt_0.2.0    
│   │   ├── default.yaml
│   │   ├── feats_stats.npy
│   │   └── snapshot_iter_2500000.pdz
│   └── hifigan_aishell3_ckpt_0.2.0.zip
└── tools
    ├── aligner
    │   ├── aishell3_model
    │   ├── aishell3_model.zip
    │   └── simple.lexicon
    ├── montreal-forced-aligner
    │   ├── bin
    │   ├── lib
    │   └── pretrained_models
    └── montreal-forced-aligner_linux.tar.gz
    ...

```

If you want to finetune English data, English label example: LJ001-0001|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition </br>
Here is an example of the first 200 data of ljspeech.

```bash
mkdir -p input && cd input
wget https://paddlespeech.bj.bcebos.com/datasets/ljspeech_mini.zip
unzip ljspeech_mini.zip
cd ../
```

When "Prepare" done. The structure of the current directory is listed below.
```text
├── input
│   ├── ljspeech_mini
│   │   ├── LJ001-0001.wav
│   │   ├── LJ001-0002.wav
│   │   ├── LJ001-0003.wav
│   │   ├── ...
│   │   ├── LJ002-0014.wav
│   │   ├── labels.txt
│   └── ljspeech_mini.zip
├── pretrained_models
│   ├── fastspeech2_vctk_ckpt_1.2.0
│   │   ├── default.yaml
│   │   ├── energy_stats.npy
│   │   ├── phone_id_map.txt
│   │   ├── pitch_stats.npy
│   │   ├── snapshot_iter_66200.pdz
│   │   ├── speaker_id_map.txt
│   │   └── speech_stats.npy
│   ├── fastspeech2_vctk_ckpt_1.2.0.zip
│   ├── hifigan_vctk_ckpt_0.2.0    
│   │   ├── default.yaml
│   │   ├── feats_stats.npy
│   │   └── snapshot_iter_2500000.pdz
│   └── hifigan_vctk_ckpt_0.2.0.zip
└── tools
    ├── aligner
    │   ├── vctk_model
    │   ├── vctk_model.zip
    │   └── cmudict-0.7b
    ├── montreal-forced-aligner
    │   ├── bin
    │   ├── lib
    │   └── pretrained_models
    └── montreal-forced-aligner_linux.tar.gz
    ...

```

### Set finetune.yaml
`conf/finetune.yaml` contains some configurations for fine-tuning. You can try various options to fine better result. The value of frozen_layers can be change according `conf/fastspeech2_layers.txt` which is the model layer of fastspeech2.

Arguments:
  - `batch_size`: finetune batch size which should be less than or equal to the number of training samples. Default: -1, means 64 which same to pretrained model
  - `learning_rate`: learning rate. Default: 0.0001
  - `num_snapshots`: number of save models. Default: -1, means 5 which same to pretrained model
  - `frozen_layers`: frozen layers. must be a list. If you don't want to frozen any layer, set []. 


## Get Started
For Chinese data finetune, execute `./run.sh`. For English data finetune, execute `./run_en.sh`. </br>
Run the command below to
1. **source path**.
2. finetune the model. 
3. synthesize wavs.
    - synthesize waveform from text file.

```bash
./run.sh
```
You can choose a range of stages you want to run, or set `stage` equal to `stop-stage` to run only one stage.

### Model Finetune

Finetune a FastSpeech2 model. 

```bash
./run.sh --stage 0 --stop-stage 5
```
`stage 5` of `run.sh` calls `local/finetune.py`, here's the complete help message.

```text
usage: finetune.py [-h] [--pretrained_model_dir PRETRAINED_MODEL_DIR]
                [--dump_dir DUMP_DIR] [--output_dir OUTPUT_DIR] [--ngpu NGPU]
                [--epoch EPOCH] [--finetune_config FINETUNE_CONFIG]

optional arguments:
  -h, --help           Show this help message and exit
  --pretrained_model_dir PRETRAINED_MODEL_DIR
                       Path to pretrained model
  --dump_dir DUMP_DIR
                       directory to save feature files and metadata
  --output_dir OUTPUT_DIR      
                       Directory to save finetune model 
  --ngpu NGPU          The number of gpu, if ngpu=0, use cpu
  --epoch EPOCH        The epoch of finetune
  --finetune_config FINETUNE_CONFIG        
                       Path to finetune config file
```

1. `--pretrained_model_dir` is the directory incluing pretrained fastspeech2_aishell3 model.
2. `--dump_dir` is the directory including audio feature and metadata.
3. `--output_dir` is the directory to save finetune model.
4. `--ngpu` is the number of gpu, if ngpu=0, use cpu
5. `--epoch` is the epoch of finetune.
6. `--finetune_config` is the path to finetune config file
 

### Synthesizing
To synthesize Chinese audio, We use [HiFiGAN with aishell3](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/voc5) as the neural vocoder.
Assume the path to the hifigan model is `./pretrained_models`. Download the pretrained HiFiGAN model from [hifigan_aishell3_ckpt_0.2.0](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip) and unzip it.

To synthesize English audio, We use [HiFiGAN with vctk](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/vctk/voc5) as the neural vocoder.
Assume the path to the hifigan model is `./pretrained_models`. Download the pretrained HiFiGAN model from [hifigan_vctk_ckpt_0.2.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip) and unzip it.


Modify `ckpt` in `run.sh` to the final model in `exp/default/checkpoints`.
```bash
./run.sh --stage 6 --stop-stage 6
```
`stage 6` of `run.sh` calls `${BIN_DIR}/../synthesize_e2e.py`, which can synthesize waveform from text file.

```text
usage: synthesize_e2e.py [-h]
                         [--am {fastspeech2_aishell3,fastspeech2_vctk}]
                         [--am_config AM_CONFIG] [--am_ckpt AM_CKPT]
                         [--am_stat AM_STAT] [--phones_dict PHONES_DICT]
                         [--tones_dict TONES_DICT]
                         [--speaker_dict SPEAKER_DICT] [--spk_id SPK_ID]
                         [--voc {pwgan_aishell3, pwgan_vctk, hifigan_aishell3, hifigan_vctk}]
                         [--voc_config VOC_CONFIG] [--voc_ckpt VOC_CKPT]
                         [--voc_stat VOC_STAT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]

Synthesize with acoustic model & vocoder

optional arguments:
  -h, --help            show this help message and exit
  --am {fastspeech2_aishell3, fastspeech2_vctk}
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
  --voc {pwgan_aishell3, pwgan_vctk, hifigan_aishell3, hifigan_vctk}
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
2. `--am_config`, `--am_ckpt`, `--am_stat`, `--phones_dict` `--speaker_dict` are arguments for acoustic model, which correspond to the 5 files in the fastspeech2 pretrained model.
3. `--voc` is vocoder type with the format {model_name}_{dataset}
4. `--voc_config`, `--voc_ckpt`, `--voc_stat` are arguments for vocoder, which correspond to the 3 files in the parallel wavegan pretrained model.
5. `--lang` is the model language, which can be `zh` or `en`.
6. `--text` is the text file, which contains sentences to synthesize.
7.  `--output_dir` is the directory to save synthesized audio files.
8. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.


### Tips
If you want to get better audio quality, you can use more audios to finetune or change configuration parameters in `conf/finetune.yaml`.</br>
More finetune results can be found on [finetune-fastspeech2-for-csmsc](https://paddlespeech.readthedocs.io/en/latest/tts/demo.html#finetune-fastspeech2-for-csmsc).</br>
The results show the effect on csmsc_mini: Freeze encoder > Non Frozen > Freeze encoder && duration_predictor.
