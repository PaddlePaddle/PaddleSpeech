# Speaker Encoder

This experiment trains a speaker encoder with speaker verification as its task. It is done as a part of the experiment of transfer learning from speaker verification to multispeaker text-to-speech synthesis, which can be found at [tacotron2_aishell3](../tacotron2_shell3). The trained speaker encoder is used to extract utterance embeddings from utterances.

## Model

The model used in this experiment is the speaker encoder with text independent speaker verification task in [GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION](https://arxiv.org/pdf/1710.10467.pdf). GE2E-softmax loss is used.

## File Structure

```text
ge2e
├── README.md
├── README_cn.md
├── audio_processor.py
├── config.py
├── dataset_processors.py
├── inference.py
├── preprocess.py
├── random_cycle.py
├── speaker_verification_dataset.py
└── train.py
```

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

## Preprocess Datasets

Multispeaker datasets are used as training data, though the transcriptions are not used. To enlarge the amount of data used for training, several multispeaker datasets are combined. The preporcessed datasets are organized in a file structure described below. The mel spectrogram of each utterance is save in `.npy` format. The dataset is 2-stratified (speaker-utterance). Since multiple datasets are combined, to avoid conflict in speaker id, dataset name is prepended to the speake ids.

```text
dataset_root
├── dataset01_speaker01/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
├── dataset01_speaker02/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
├── dataset02_speaker01/
│   ├── utterance01.npy
│   ├── utterance02.npy
│   └── utterance03.npy
└── dataset02_speaker02/
    ├── utterance01.npy
    ├── utterance02.npy
    └── utterance03.npy
```

Run the command to preprocess datasets.

```bash
python preprocess.py --datasets_root=<datasets_root> --output_dir=<output_dir> --dataset_names=<dataset_names>
```

Here `--datasets_root` is the directory that contains several extracted dataset; `--output_dir` is the directory to save the preprocessed dataset; `--dataset_names` is the dataset to preprocess. If there are multiple datasets in `--datasets_root` to preprocess, the names can be joined with comma. Currently supported dataset names are  librispeech_other, voxceleb1, voxceleb2, aidatatang_200zh and magicdata.

## Training

When preprocessing is done, run the command below to train the mdoel.

```bash
python train.py --data=<data_path> --output=<output> --device="gpu" --nprocs=1
```

- `--data` is the path to the preprocessed dataset.
- `--output` is the directory to save results，usually a subdirectory of `runs`.It contains visualdl log files, text log files, config file and a `checkpoints` directory, which contains parameter file and optimizer state file. If `--output` already has some training results in it, the most recent parameter file and optimizer state file is loaded before training.
- `--device` is the device type to run the training, 'cpu' and 'gpu' are supported.
- `--nprocs` is the number of replicas to run in multiprocessing based parallel training。Currently multiprocessing based parallel training is only enabled when using 'gpu' as the devicde. `CUDA_VISIBLE_DEVICES` can be used to specify visible devices with cuda.

Other options are described below.

- `--config` is a `.yaml` config file used to override the default config(which is coded in `config.py`).
- `--opts` is command line options to further override config files. It should be the last comman line options passed with multiple key-value pairs separated by spaces.
- `--checkpoint_path` specifies the checkpoiont to load before training, extension is not included. A parameter file ( `.pdparams`) and an optimizer state file ( `.pdopt`) with the same name is used. This option has a higher priority than auto-resuming from the `--output` directory.

## Pretrained Model

The pretrained model is first trained to 1560k steps at Librispeech-other-500 and voxceleb1. Then trained at aidatatang_200h and magic_data to 3000k steps.

Download URL [ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip).

## Inference

When training is done, run the command below to generate utterance embedding for each utterance in a dataset.

```bash
python inference.py --input=<input> --output=<output> --checkpoint_path=<checkpoint_path> --device="gpu"
```

`--input` is the path of the dataset used for inference.

`--output` is the directory to save the processed results. It has the same file structure as the input dataset. Each utterance in the dataset has a corrsponding utterance embedding file in `*.npy` format.

`--checkpoint_path` is the path of the checkpoint to use, extension not included.

`--pattern` is the wildcard pattern to filter audio files for inference, defaults to `*.wav`.

`--device` and `--opts` have the same meaning as in the training script.

## References

1. [Generalized End-to-end Loss for Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf)
2. [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf)
