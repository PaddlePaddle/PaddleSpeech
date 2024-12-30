# DeepSpeech2 offline/online ASR with Librispeech
This example contains code used to train a DeepSpeech2 offline or online model with [[Librispeech dataset](http://www.openslr.org/resources/12)](http://www.openslr.org/resources/33)
## Overview
All the scripts you need are in the `run.sh`. There are several stages in the `run.sh`, and each stage has its function.
| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the dataset <br>       (2) Calculate the CMVN of the train dataset <br>       (3) Get the vocabulary file <br>       (4) Get the manifest files of the train, development and test dataset |
| 1     | Train the model                                              |
| 2     | Get the final model by averaging the top-k models, set k = 1 means to choose the best model |
| 3     | Test the final model performance                             |
| 4     | Export the static graph model     |
| 5      | Test the static graph model      |
| 6     | Infer the single audio file                                  |

You can choose to run a range of stages by setting the `stage` and `stop_stage `. 

For example, if you want to execute the code in stage 2 and stage 3, you can run this script:
```bash
bash run.sh --stage 2 --stop_stage 3
```
Or you can set `stage` equal to `stop-stage` to only run one stage.
For example, if you only want to run `stage 0`, you can use the script below:
```bash
bash run.sh --stage 0 --stop_stage 0
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
`stop stage` denotes the number of stages you want to end at in the experiments. 
`conf_path` denotes the config path of the model.
`avg_num` denotes the number K of top-K models you want to average to get the final model.
`model_type` denotes the model type: offline or online
`audio file` denotes the file path of the single file you want to infer in stage 6
`ckpt` denotes the checkpoint prefix of the model, e.g. "deepspeech2"

You can set the local variables (except `ckpt`)  when you use the `run.sh`

For example, you can set the `gpus` and `avg_num` when you use the command line.:
```bash
bash run.sh --gpus 0,1 --avg_num 1
```
## Stage 0: Data processing
To use this example, you need to process data firstly and  you can use stage 0 in the `run.sh` to do this. The code is shown below:
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
source path.sh
bash ./local/data.sh
```
After processing the data, the `data` directory will look like this:
```bash
data/
|-- dev.meta
|-- lang_char
|   `-- vocab.txt
|-- manifest.dev
|-- manifest.dev.raw
|-- manifest.test
|-- manifest.test.raw
|-- manifest.train
|-- manifest.train.raw
|-- mean_std.json
|-- test.meta
`-- train.meta
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
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
```
## Stage 2:  Top-k Models Averaging
After training the model,  we need to get the final model for testing and inference. In every epoch, the model checkpoint is saved, so we can choose the best model from them based on the validation loss or we can sort them and average the parameters of the top-k models to get the final model.  We can use stage 2 to do this, and the code is shown below:
```bash
 if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     # avg n best model
     avg.sh best exp/${ckpt}/checkpoints ${avg_num}
 fi
```
The `avg.sh` is in the `../../../utils/` which is define in the `path.sh`.
If you want to get the final model,  you can use the script below to execute stage 0, stage 1, and stage 2:
```bash
bash run.sh --stage 0 --stop_stage 2
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
avg.sh best exp/deepspeech2/checkpoints 1
```
## Stage 3: Model Testing
The test stage is to evaluate the model performance. The code of the test stage is shown below:
```bash
 if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
     # test ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
 fi
```
If you want to train a model and test it,  you can use the script below to execute stage 0, stage 1,  stage 2, and stage 3 :
```bash
bash run.sh --stage 0 --stop_stage 3
```
or you can run these scripts in the command line (only use CPU).
```bash
source path.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/deepspeech2.yaml deepspeech2
avg.sh best exp/deepspeech2/checkpoints 1
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/deepspeech2.yaml conf/tuning/decode.yaml exp/deepspeech2/checkpoints/avg_1
```
## Stage 4: Static graph model Export
This stage is to transform dygraph to static graph.
```bash
 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
     # export ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} exp/${ckpt}/checkpoints/${avg_ckpt}.jit ${model_type}
 fi
```
If you already have a dynamic graph model, you can run this script:
```bash
source path.sh
./local/export.sh deepspeech2.yaml exp/deepspeech2/checkpoints/avg_1 exp/deepspeech2/checkpoints/avg_1.jit offline
```
## Stage 5: Static graph Model Testing
Similar to stage 3, the static graph model can also be tested.
```bash
 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
     # test export ckpt avg_n
     CUDA_VISIBLE_DEVICES=0 ./local/test_export.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt}.jit ${model_type}|| exit -1
 fi
```
If you already have exported the static graph, you can run this script:
```bash
CUDA_VISIBLE_DEVICES= ./local/test_export.sh conf/deepspeech2.yaml exp/deepspeech2/checkpoints/avg_1.jit offline
```
## Stage 6: Single Audio File Inference
In some situations, you want to use the trained model to do the inference for the single audio file. You can use stage  5. The code is shown below
```bash
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
     # test a single .wav file
     CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${model_type} ${audio_file}
 fi
```
You can download the audio demo:
```bash
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/en/demo_002_en.wav -P data/
```
You can train a model by yourself, then you need to prepare an audio file or use the audio demo above, please confirm the sample rate of the audio is 16K. You can get the result of the audio demo by running the script below.
```bash
CUDA_VISIBLE_DEVICES= ./local/test_wav.sh conf/deepspeech2.yaml conf/tuning/decode.yaml exp/deepspeech2/checkpoints/avg_1 data/demo_002_en.wav
```
