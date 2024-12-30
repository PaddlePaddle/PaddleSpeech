
# Transformer/Conformer ST0 with TED_En_Zh
This example contains code used to train a Transformer or [Conformer](http://arxiv.org/abs/2008.03802) model with TED_EN_Zh
## Overview
All the scripts you need are in `run.sh`. There are several stages in `run.sh`, and each stage has its function.

You need to download TED_En_Zh dataset by yourself.

| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Calculate the CMVN of the train dataset <br>       (2) Get the vocabulary file <br>       (3) Get the manifest files of the train, development and test dataset<br> |
| 1     | Train the model                                              |
| 2     | Get the final model by averaging the top-k models, set k = 1 means to choose the best model |
| 3     | Test the final model performance                             |

You can choose to run a range of stages by setting `stage` and `stop_stage `. 

For example, if you want to execute the code in stage 2 and stage 3, you can run this script:
```bash
bash run.sh --stage 2 --stop_stage 3
```
Or you can set `stage` equal to `stop-stage` to only run one stage.
For example, if you only want to run ```stage 0```, you can use the script below:
```bash
bash run.sh --stage 0 --stop_stage 0
```
The document below will describe the scripts in `run.sh` in detail.

## The Environment Variables
The path.sh contains the environment variables. 
```bash
source path.h
```
This script needs to be run first. And another script is also needed:
```bash
source ${MAIN_ROOT}/utils/parse_options.sh
```
It will support the way of using `--variable value` in the shell scripts.

## The Local Variables
Some local variables are set in `run.sh`. 
`gpus` denotes the GPU number you want to use. If you set `gpus=`, it means you only use CPU. 
`stage` denotes the number of stages you want to start from in the experiments.
`stop_stage` denotes the number of stages you want to end at in the experiments. 
`conf_path`denotes the config path of the model.
`data_path` denotes the path of the dataset.
`avg_num`denotes the number K of top-K models you want to average to get the final model.
`ckpt` denotes the checkpoint prefix of the model, e.g. "transformer_mtl_noam"

You can set the local variables (except `ckpt`) when you use `run.sh`

For example, you can set the `gpus` and `avg_num` when you use the command line.:
```bash
bash run.sh --gpus 0,1 --avg_num 5
```
## Stage 0: Data Processing
To use this example, you need to process data firstly and you can use stage 0 in ```run.sh```to do this. The code is shown below:
```bash
 if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     # prepare data
     bash ./local/data.sh || exit -1
 fi
```
Stage 0 is for processing the data.
If you only want to process the data. You can run
```bash
bash run.sh --stage 0 --stop_stage 0
```
You can also just run these scripts in your command line.
```bash
source path.h
bash ./local/data.sh
```
## Stage 1: Model Training
If you want to train the model. you can use stage 1 in ```run.sh```. The code is shown below. 
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     # train model, all `ckpt` under `exp` dir
     CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${ckpt}
 fi
```
If you want to train the model, you can use the script below to execute stage 0 and stage 1:
```bash
bash run.sh --stage 0 --stop_stage 1
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.h
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer_mtl_noam.yaml transformer_mtl_noam
```
## Stage 2: Top-k Models Averaging
After training the model,  we need to get the final model for testing and inference. In every epoch, the model checkpoint is saved, so we can choose the best model from them based on the validation loss or we can sort them and average the parameters of the top-k models to get the final model.  We can use stage 2 to do this, and the code is shown below:
```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    avg.sh best exp/${ckpt}/checkpoints ${avg_num}
fi
```
The ```avg.sh```is in the ```../../../utils/```which is define in the ```path.sh```.
If you want to get the final model, you can use the script below to execute stage 0, stage 1, and stage 2:
```bash
bash run.sh --stage 0 --stop_stage 2
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.h
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer_mtl_noam.yaml transformer_mtl_noam
avg.sh best exp/transformer_mtl_noam/checkpoints 5
```
## Stage 3: Model Testing
The  stage 3 is to evaluate the model performance. The code of this stage is shown below:
```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```
If you want to train a model and test it, you can use the script below to execute stage 0, stage 1, stage 2, and stage 3 :
```bash
bash run.sh --stage 0 --stop_stage 3
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.h
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer_mtl_noam.yaml transformer_mtl_noam
avg.sh latest exp/transformer_mtl_noam/checkpoints 5
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/transformer_mtl_noam.yaml conf/tuning/decode.yaml exp/transformer_mtl_noam/checkpoints/avg_5
```
The performance of the released models are shown below:
### Transformer
| Model               | Params | Config                           | Char-BLEU |
| ------------------- | ------ | -------------------------------- | --------- |
| Transformer+ASR MTL | 50.26M | conf/transformer_joint_noam.yaml | 17.38     |
