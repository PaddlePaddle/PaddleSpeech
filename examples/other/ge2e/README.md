# Speaker Encoder
This experiment trains a speaker encoder with speaker verification as its task. It is done as a part of the experiment of transfer learning from speaker verification to multispeaker text-to-speech synthesis, which can be found at [examples/aishell3/vc0](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/examples/aishell3/vc0). The trained speaker encoder is used to extract utterance embeddings from utterances.
## Model
The model used in this experiment is the speaker encoder with text independent speaker verification task in [GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION](https://arxiv.org/pdf/1710.10467.pdf). GE2E-softmax loss is used.

## Download Datasets
Currently supported datasets are  Librispeech-other-500, VoxCeleb, VoxCeleb2,ai-datatang-200zh, magicdata, which can be downloaded from corresponding webpage.

1. Librispeech/train-other-500
   An English multispeaker dataset，[URL](https://www.openslr.org/resources/12/train-other-500.tar.gz)，only the `train-other-500` subset is used.
2. VoxCeleb1
   An English multispeaker dataset，[URL](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) , Audio Files from Dev A to Dev D should be downloaded, combined and extracted.
3. VoxCeleb2
   An English multispeaker dataset，[URL](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) , Audio Files from Dev A to Dev H should be downloaded, combined and extracted.
4. Aidatatang-200zh
   A Mandarin Chinese multispeaker dataset ，[URL](https://www.openslr.org/62/) .
5. magicdata
   A Mandarin Chinese multispeaker dataset ，[URL](https://www.openslr.org/68/) .

If you want to use other datasets, you can also download and preprocess it as long as it meets the requirements described below.

## Get Started

```bash
./run.sh
```

### Preprocess Datasets
`./local/preprocess.sh` calls `${BIN_DIR}/preprocess.py`.
```bash
./local/preprocess.sh ${datasets_root} ${preprocess_path} ${dataset_names}
```
Assume datasets_root is `~/datasets/GE2E`, and it has the follow structure（We only use `train-other-500` for simplicity）:
```Text
GE2E
├── LibriSpeech
└── (other datasets)
```
Multispeaker datasets are used as training data, though the transcriptions are not used. To enlarge the amount of data used for training, several multispeaker datasets are combined. The preporcessed datasets are organized in a file structure described below. The mel spectrogram of each utterance is save in `.npy` format. The dataset is 2-stratified (speaker-utterance). Since multiple datasets are combined, to avoid conflict in speaker id, dataset name is prepended to the speake ids.

```text
dataset_root
├── dataset01_speaker01/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
├── dataset01_speaker02/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
├── dataset02_speaker01/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
└── dataset02_speaker02/
    ├── utterance01.npy
    ├── utterance02.npy
    └── utterance03.npy
```
In `${BIN_DIR}/preprocess.py`:
1. `--datasets_root` is the directory that contains several extracted dataset
2.  `--output_dir` is the directory to save the preprocessed dataset
3.  `--dataset_names` is the dataset to preprocess. If there are multiple datasets in `--datasets_root` to preprocess, the names can be joined with comma. Currently supported dataset names are  librispeech_other, voxceleb1, voxceleb2, aidatatang_200zh and magicdata.

### Train the model
`./local/train.sh` calls `${BIN_DIR}/train.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path}
```
In `${BIN_DIR}/train.py`:
1. `--data` is the path to the preprocessed dataset.
2. `--output` is the directory to save results，usually a subdirectory of `runs`.It contains visualdl log files, text log files, config file and a `checkpoints` directory, which contains parameter file and optimizer state file. If `--output` already has some training results in it, the most recent parameter file and optimizer state file is loaded before training.
3. `--device` is the device type to run the training, 'cpu' and 'gpu' are supported.
4. `--nprocs` is the number of replicas to run in multiprocessing based parallel training。Currently multiprocessing based parallel training is only enabled when using 'gpu' as the devicde.
5. `CUDA_VISIBLE_DEVICES` can be used to specify visible devices with cuda.

Other options are described below.

- `--config` is a `.yaml` config file used to override the default config(which is coded in `config.py`).
- `--opts` is command line options to further override config files. It should be the last comman line options passed with multiple key-value pairs separated by spaces.
- `--checkpoint_path` specifies the checkpoiont to load before training, extension is not included. A parameter file ( `.pdparams`) and an optimizer state file ( `.pdopt`) with the same name is used. This option has a higher priority than auto-resuming from the `--output` directory.

###  Inference
When training is done, run the command below to generate utterance embedding for each utterance in a dataset.
`./local/inference.sh` calls `${BIN_DIR}/inference.py`.
```bash
CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${infer_input} ${infer_output} ${train_output_path} ${ckpt_name}
```
In `${BIN_DIR}/inference.py`:
1. `--input` is the path of the dataset used for inference.
2. `--output` is the directory to save the processed results. It has the same file structure as the input dataset. Each utterance in the dataset has a corrsponding utterance embedding file in `*.npy` format.
3. `--checkpoint_path` is the path of the checkpoint to use, extension not included.
4. `--pattern` is the wildcard pattern to filter audio files for inference, defaults to `*.wav`.
5. `--device` and `--opts` have the same meaning as in the training script.

## Pretrained Model
The pretrained model is first trained to 1560k steps at Librispeech-other-500 and voxceleb1. Then trained at aidatatang_200h and magic_data to 3000k steps.

Download URL [ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip).

## References

1. [Generalized End-to-end Loss for Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf)
2. [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf)
