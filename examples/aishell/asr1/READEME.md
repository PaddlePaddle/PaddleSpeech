# Conformer ASR with Aishell

This example contains code used to train a [Conformer](http://arxiv.org/abs/2008.03802) model with [Aishell dataset](http://www.openslr.org/resources/33)

## Overview

All the scirpt you need is in the ```run.sh```. There are several stages in the ```run.sh```, and each stage has it's function.

| Stage | Function                                                     |
| :---- | :----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the dataset <br>       (2) Caculate the CMVN of the train dataset <br>       (3) Get the vocabulary file <br>       (4) Get the manifest files of the train, development and test dataset |
| 1     | Train the model                                              |
| 2     | Get the final model by average the top-k model , set k = 1 means choose the best model |
| 3     | Test the final model performance                             |
| 4     | Get ctc alignment of test data using the final model         |
| 5     | Infer the single audio file                                  |
| 51    | (Not supported at now) Transform the dynamic graph model to static graph model |
| 101   | (Need further installation) Train language model and Build TLG |

You can choose to run a range of  stages by set the ```stage``` and ```stop_stage ``` . 

For example , if you want to execute the code in stage 2 and stage 3, you can run this script:

```bash
bash run.sh --stage 2 --stop_stage 3
```
Or you can set ```stage``` equal to ```stop-stage``` to only run one stage.
For example, if you only want to run ```stage 0```, you can use the script below:

```bash
bash run.sh --stage 0 --stop_stage 0
```



The document below will decribe the scripts in the ```run.sh``` in detail.

## The environment variables

The path.sh contains the environment variable. 
```bash
source path.sh
```
This script needs to be run firstly.  

And another script is also needed:

```bash
source ${MAIN_ROOT}/utils/parse_options.sh
```

It will support the way of using```--varibale value``` in the shell scripts.



## The local variables

Some local variables are set in the ```run.sh```. 
```gpus``` denotes the GPU number you want to use. If you set ```gpus=```,  it means you only use CPU. 

```stage``` denotes  the number of stage you want to start from in the expriments.
```stop stage```denotes the number of stage you want to end at in the expriments. 

```conf_path``` denotes the config path of the model.

```avg_num``` denotes the number K of top-K model you want to average to get the final model.

```ckpt``` denotes the checkpoint prefix of the model, e.g. "conformer"

```audio file``` denotes the file path of the single file you want to infer in stage 6

You can set the local variables when you use the ```run.sh```

For example, you can set the ```gpus``` and ``avg_num`` when you use the command line.:

```bash
bash run.sh --gpus 0,1 --avg_num 20
```



## Stage 0: Data processing

To use this example, you need to process data firstly and  you can use the stage 0 in the ```run.sh``` to do this. The code is shown below:
```bash
 if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
     # prepare data
     bash ./local/data.sh || exit -1
 fi
```

The stage 0 is for processing the data.

If you only want to process the data. You can run

```bash
bash run.sh --stage 0 --stop_stage 0
```

You can also just run these scripts in your command line.

```bash
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
bash ./local/data.sh
```

Aftre processing the data, the ``data`` directory will be look like this:

```bash
data/
├── dev.meta
├── manifest.dev.raw
├── manifest.test.raw
├── manifest.train.raw
├── mean_std.json
├── test.meta
├── train.meta
└── vocab.txt
```



## Stage 1: Model training

If you want to train the model. you can use the stage 1 in the ```run.sh```. The code is shown below. 
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     # train model, all `ckpt` under `exp` dir
     CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path}  ${ckpt}
 fi
```

If you want to train the model, you can use the script below to execute the stage 0 and stage 1:
```bash
bash run.sh --stage 0 --stop_stage 1
```
or you can run these scripts in command line (only use CPU).
```bash
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml  conformer
```



## Stage 2:  Top-k model averaging

After training the model,  we need to get the final model for test and infer. In every epoch, the model checkpoint is saved , so we can choose the best model from them based on the validation loss or we can sort them and average the top-k model parameters to get the final model.  We can use the stage 2 to do this, and the code is shown below:
```bash
 if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     # avg n best model
     avg.sh best exp/${ckpt}/checkpoints ${avg_num}
 fi
```
The ```avg.sh``` is in the ```../../../utils/``` which is define in the ```path.sh```.
If you want to get the final model,  you can use the script below to execute the stage 0, stage 1, and stage 2:

```bash
bash run.sh --stage 0 --stop_stage 2
```

or you can run these scripts in command line (only use CPU).
```bash
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
avg.sh best exp/conformer/checkpoints 20
```



## Stage 3: Model Testing

To know the preformence of the model, test stage is needed. The code of test stage is shown below:

```bash
 if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
     # test ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
 fi
```

If you want to train a model and test it,  you can use the script below to execute the stage 0, stage 1,  stage 2, and stage 3 :

```bash
bash run.sh --stage 0 --stop_stage 3
```

or you can run these scripts in command line (only use CPU).

```bash
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
avg.sh best exp/conformer/checkpoints 20
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/conformer.yaml exp/conformer/checkpoints/avg_20
```



## Stage 4: CTC alignment 

If you want to get the alignment between the audio and the text, you can use the ctc alignment. The code of this stage is shown below:

```bash
 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
     # ctc alignment of test data
     CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
 fi
```

If you want to train the model, test it and do the alignment,  you can use the script below to execute the stage 0, stage 1,  stage 2, and stage 3 :

```bash
bash run.sh --stage 0 --stop_stage 4
```

or if you only need to train a model and do the alignment, you can use these scripts to escape the stage 3(test stage):

```bash
bash run.sh --stage 0 --stop_stage 2
bash run.sh --stage 4 --stop_stage 4
```

or you can also use these scripts in command line (only use CPU).

```bash
source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
avg.sh best exp/conformer/checkpoints 20
# test stage is optional
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/conformer.yaml exp/conformer/checkpoints/avg_20
CUDA_VISIBLE_DEVICES= ./local/align.sh conf/conformer.yaml exp/conformer/checkpoints/avg_20
```



## Stage 5: Single audio file inference

In some situation, you want to use the trained model to do the inference for the single audio file. You can use the stage  5. The code is shown below

```bash
 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
     # test a single .wav file
     CUDA_VISIBLE_DEVICES=0 ./local/test_hub.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
 fi
```

you can train the model by yourself using ```bash run.sh --stage 0 --stop_stage 3```， or you can download the pretrained model by the script below:

```
wget https://deepspeech.bj.bcebos.com/release2.1/aishell/s1/aishell.release.tar.gz
tar aishell.release.tar.gz
```

You need to prepare an audio file, please confirme the sample rate of the audio is 16K.  Assume the path of the audio file is ```data/test_audio.wav```,  you can get the result by runing the script below.

```bash
CUDA_VISIBLE_DEVICES= ./local/test_hub.sh conf/conformer.yaml exp/conformer/checkpoints/avg_20 data/test_audio.wav
```



## Stage 51: Static model transforming（not supported at now）

To transform the dynamic model to static model，stage 51 can be used. The code of this stage is shown below:

```bash
 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
     # export ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit
 fi
```

It is not supported at now, so we set a large stage number for this stage.



## Stage: 101 Language model training and TLG building (Need further installation! )

You need to install the kaldi and srilm to use the stage 101,  it is used for train language model and build TLG. To do further installation, you need to do these:

```bash
# go to the root of the repo
cd ../../../ 
# Do the further installation
pip install -e .
cd tools
bash extras/install_openblas.sh
bash extras/install_kaldi.sh
```

You need to be patient, since installing the kaldi takes some time.


