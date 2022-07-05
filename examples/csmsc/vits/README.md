# VITS with CSMSC
This example contains code used to train a [VITS](https://arxiv.org/abs/2106.06103) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).

## Dataset
### Download and Extract
Download CSMSC from it's [Official Website](https://test.data-baker.com/data/index/source).

### Get MFA Result and Extract
We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get phonemes for VITS, the durations of MFA are not needed here.
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
│   ├── norm
│   └── raw
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
The dataset is split into 3 parts, namely `train`, `dev`, and` test`, each of which contains a `norm` and `raw` subfolder. The raw folder contains wave and linear spectrogram of each utterance, while the norm folder contains normalized ones. The statistics used to normalize features are computed from the training set, which is located in `dump/train/feats_stats.npy`.

Also, there is a `metadata.jsonl` in each subfolder. It is a table-like file that contains phones, text_lengths, feats, feats_lengths, the path of linear spectrogram features, the path of raw waves, speaker, and the id of each utterance.

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

Train a VITS model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file to overwrite default config.
  --train-metadata TRAIN_METADATA
                        training data.
  --dev-metadata DEV_METADATA
                        dev data.
  --output-dir OUTPUT_DIR
                        output dir.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --phones-dict PHONES_DICT
                        phone vocabulary file.
```
1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are saved in `checkpoints/` inside this directory.
4. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.
5. `--phones-dict` is the path of the phone vocabulary file.

### Synthesizing

`./local/synthesize.sh` calls `${BIN_DIR}/synthesize.py`, which can synthesize waveform from `metadata.jsonl`.

```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize.py [-h] [--config CONFIG] [--ckpt CKPT]
                     [--phones_dict PHONES_DICT] [--ngpu NGPU]
                     [--test_metadata TEST_METADATA] [--output_dir OUTPUT_DIR]

Synthesize with VITS

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config of VITS.
  --ckpt CKPT           Checkpoint file of VITS.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --ngpu NGPU           if ngpu == 0, use cpu.
  --test_metadata TEST_METADATA
                        test metadata.
  --output_dir OUTPUT_DIR
                        output dir.
```
`./local/synthesize_e2e.sh` calls `${BIN_DIR}/synthesize_e2e.py`, which can synthesize waveform from text file.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name}
```
```text
usage: synthesize_e2e.py [-h] [--config CONFIG] [--ckpt CKPT]
                         [--phones_dict PHONES_DICT] [--lang LANG]
                         [--inference_dir INFERENCE_DIR] [--ngpu NGPU]
                         [--text TEXT] [--output_dir OUTPUT_DIR]

Synthesize with VITS

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Config of VITS.
  --ckpt CKPT           Checkpoint file of VITS.
  --phones_dict PHONES_DICT
                        phone vocabulary file.
  --lang LANG           Choose model language. zh or en
  --inference_dir INFERENCE_DIR
                        dir to save inference models
  --ngpu NGPU           if ngpu == 0, use cpu.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line.
  --output_dir OUTPUT_DIR
                        output dir.
```
1. `--config`, `--ckpt`, and `--phones_dict` are arguments for acoustic model, which correspond to the 3 files in the VITS pretrained model.
2. `--lang` is the model language, which can be `zh` or `en`.
3. `--test_metadata` should be the metadata file in the normalized subfolder of `test`  in the `dump` folder.
4. `--text` is the text file, which contains sentences to synthesize.
5. `--output_dir` is the directory to save synthesized audio files.
6. `--ngpu` is the number of gpus to use, if ngpu == 0, use cpu.

## Pretrained Model

The pretrained model can be downloaded here:

- [vits_csmsc_ckpt_1.1.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/vits/vits_csmsc_ckpt_1.1.0.zip) (add_blank=true)

VITS checkpoint contains files listed below.
```text
vits_csmsc_ckpt_1.1.0
├── default.yaml              # default config used to train vitx
├── phone_id_map.txt          # phone vocabulary file when training vits
└── snapshot_iter_350000.pdz  # model parameters and optimizer states
```

ps: This ckpt is not good enough, a better result is training

You can use the following scripts to synthesize for `${BIN_DIR}/../sentences.txt` using pretrained VITS.

```bash
source path.sh
add_blank=true

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
    --config=vits_csmsc_ckpt_1.1.0/default.yaml \
    --ckpt=vits_csmsc_ckpt_1.1.0/snapshot_iter_350000.pdz \
    --phones_dict=vits_csmsc_ckpt_1.1.0/phone_id_map.txt \
    --output_dir=exp/default/test_e2e \
    --text=${BIN_DIR}/../sentences.txt \
    --add-blank=${add_blank} 
```
