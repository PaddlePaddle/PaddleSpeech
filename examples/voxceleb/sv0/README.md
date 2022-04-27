# ECAPA-TDNN with VoxCeleb
This example contains code used to train a ECAPA-TDNN model with [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#about)

## Overview
All the scripts you need are in the `run.sh`. There are several stages in the `run.sh`, and each stage has its function.
| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the VoxCeleb1 dataset <br>       (2) Download the VoxCeleb2 dataset  <br>       (3) Convert the VoxCeleb2 m4a to wav format <br>       (4) Get the manifest files of the train, development and test dataset <br> (5) Download the RIR Noise dataset and Get the noise manifest files for augmentation |
| 1     | Train the model                                              |
| 2     | Test the speaker verification with VoxCeleb trial|

You can choose to run a range of stages by setting the `stage` and `stop_stage `. 

For example, if you want to execute the code in stage 1 and stage 2, you can run this script:
```bash
bash run.sh --stage 1 --stop_stage 2
```
Or you can set `stage` equal to `stop-stage` to only run one stage.
For example, if you only want to run `stage 0`, you can use the script below:
```bash
bash run.sh --stage 1 --stop_stage 1
```
The document below will describe the scripts in the `run.sh` in detail.
## The environment variables
The path.sh contains the environment variable. 
```bash
source path.sh
```
This script needs to be run first.  

And another script is also needed:
```bash
source ${MAIN_ROOT}/utils/parse_options.sh
```
It will support the way of using `--variable value` in the shell scripts.

## The local variables
Some local variables are set in the `run.sh`. 
`gpus` denotes the GPU number you want to use. If you set `gpus=`,  it means you only use CPU. 
`stage` denotes the number of the stage you want to start from in the experiments.
`stop stage` denotes the number of the stage you want to end at in the experiments. 
`conf_path` denotes the config path of the model.
`exp_dir` denotes the experiment directory, e.g. "exp/ecapa-tdnn-vox12-big/"

You can set the local variables when you use the `run.sh`

For example, you can set the `gpus` when you use the command line.:
```bash
bash run.sh --gpus 0,1 
```
## Stage 0: Data processing
To use this example, you need to process data firstly and you can use stage 0 in the `run.sh` to do this. The code is shown below:

```bash
 if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     # prepare data
     bash ./local/data.sh || exit -1
 fi
```
Stage 0 is for processing the data. If you only want to process the data. You can run
```bash
bash run.sh --stage 0 --stop_stage 0
```
You can also just run these scripts in your command line.
```bash
source path.sh
bash ./local/data.sh
```
After processing the data, the `data` directory will look like this:
```bash
data/
├── rir_noise
│   ├── csv
│   │   ├── noise.csv
│   │   └── rir.csv
│   ├── manifest.pointsource_noises
│   ├── manifest.real_rirs_isotropic_noises
│   └── manifest.simulated_rirs
├── vox
│   ├── csv
│   │   ├── dev.csv
│   │   ├── enroll.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── meta
│       └── label2id.txt
└── vox1
    ├── list_test_all2.txt
    ├── list_test_all.txt
    ├── list_test_hard2.txt
    ├── list_test_hard.txt
    ├── manifest.dev
    ├── manifest.test
    ├── veri_test2.txt
    ├── veri_test.txt
    ├── voxceleb1.dev.meta
    └── voxceleb1.test.meta
```
## Stage 1: Model training
If you want to train the model. you can use stage 1 in the `run.sh`. The code is shown below. 
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     # train model, all `ckpt` under `exp` dir
     CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path}  ${ckpt}
 fi
```
If you want to train the model, you can use the script below to execute stage 0 and stage 1:
```bash
bash run.sh --stage 0 --stop_stage 1
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.sh
bash ./local/data.sh ./data/ conf/ecapa_tdnn.yaml
CUDA_VISIBLE_DEVICES= ./local/train.sh ./data/ exp/ecapa-tdnn-vox12-big/ conf/ecapa_tdnn.yaml
```
## Stage 2: Model Testing
The test stage is to evaluate the model performance. The code of the test stage is shown below:
```bash
 if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     # test ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${dir} ${exp_dir} ${conf_path} || exit -1
 fi
```
If you want to train a model and test it,  you can use the script below to execute stage 0, stage 1 and stage 2:
```bash
bash run.sh --stage 0 --stop_stage 2
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.sh
bash ./local/data.sh ./data/ conf/ecapa_tdnn.yaml
CUDA_VISIBLE_DEVICES= ./local/train.sh ./data/ exp/ecapa-tdnn-vox12-big/ conf/ecapa_tdnn.yaml
CUDA_VISIBLE_DEVICES= ./local/test.sh ./data/ exp/ecapa-tdnn-vox12-big/ conf/ecapa_tdnn.yaml
```

## 3: Pretrained Model
You can get the pretrained models from [this](../../../docs/source/released_model.md).

using the `tar` scripts to unpack the model and then you can use the script to test the model.

For example:
```
wget https://paddlespeech.bj.bcebos.com/vector/voxceleb/sv0_ecapa_tdnn_voxceleb12_ckpt_0_2_0.tar.gz
tar -xvf sv0_ecapa_tdnn_voxceleb12_ckpt_0_2_0.tar.gz
source path.sh
# If you have processed the data and get the manifest file， you can skip the following 2 steps

CUDA_VISIBLE_DEVICES= bash ./local/test.sh ./data sv0_ecapa_tdnn_voxceleb12_ckpt_0_1_2/model/ conf/ecapa_tdnn.yaml
```
The performance of the released models are shown in [this](./RESULTS.md)
