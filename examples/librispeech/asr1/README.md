# Transformer/Conformer ASR with Librispeech
This example contains code used to train [u2](https://arxiv.org/pdf/2012.05481.pdf) model (Transformer or [Conformer](https://arxiv.org/pdf/2005.08100.pdf) model) with [Librispeech dataset](http://www.openslr.org/resources/12)
## Overview
All the scripts you need are in `run.sh`. There are several stages in `run.sh`, and each stage has its function.
| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the dataset <br>       (2) Calculate the CMVN of the train dataset <br>       (3) Get the vocabulary file <br>       (4) Get the manifest files of the train, development and test dataset<br>       (5) Get the sentencepiece model |
| 1     | Train the model                                              |
| 2     | Get the final model by averaging the top-k models, set k = 1 means to choose the best model |
| 3     | Test the final model performance                             |
| 4     | Get ctc alignment of test data using the final model         |
| 5     | Infer the single audio file                                  |

You can choose to run a range of stages by setting `stage` and `stop_stage `. 

For example, if you want to execute the code in stage 2 and stage 3, you can run this script:
```bash
bash run.sh --stage 2 --stop_stage 3
```
Or you can set `stage` equal to `stop-stage` to only run one stage.
For example, if you only want to run `stage 0`, you can use the script below:
```bash
bash run.sh --stage 0 --stop_stage 0
```
The document below will describe the scripts in `run.sh` in detail.
## The Environment Variables
The path.sh contains the environment variables. 
```bash
. ./path.sh
. ./cmd.sh
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
`stop stage` denotes the number of the stage you want to end at in the experiments. 
`conf_path` denotes the config path of the model.
`avg_num` denotes the number K of top-K models you want to average to get the final model.
`audio file` denotes the file path of the single file you want to infer in stage 5
`ckpt` denotes the checkpoint prefix of the model, e.g. "conformer"

You can set the local variables (except `ckpt`) when you use `run.sh`

For example, you can set the `gpus` and `avg_num` when you use the command line:
```bash
bash run.sh --gpus 0,1 --avg_num 20
```
## Stage 0: Data Processing
To use this example, you need to process data firstly and you can use stage 0 in `run.sh` to do this. The code is shown below:
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
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
```
After processing the data, the `data` directory will look like this:
```bash
data/
|-- dev.meta
|-- lang_char
|   `-- bpe_unigram_5000.model
|   `-- bpe_unigram_5000.vocab
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
## Stage 1: Model Training
If you want to train the model. you can use stage 1 in `run.sh`. The code is shown below. 
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
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
```
## Stage 2: Top-k Models Averaging
After training the model, we need to get the final model for testing and inference. In every epoch, the model checkpoint is saved, so we can choose the best model from them based on the validation loss or we can sort them and average the parameters of the top-k models to get the final model. We can use stage 2 to do this, and the code is shown below:
```bash
 if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     # avg n best model
     avg.sh best exp/${ckpt}/checkpoints ${avg_num}
 fi
```
The `avg.sh` is in the `../../../utils/` which is define in the `path.sh`.
If you want to get the final model, you can use the script below to execute stage 0, stage 1, and stage 2:
```bash
bash run.sh --stage 0 --stop_stage 2
```
or you can run these scripts in the command line (only use CPU).

```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
avg.sh best exp/conformer/checkpoints 20
```
## Stage 3: Model Testing
The test stage is to evaluate the model performance. The code of test stage is shown below:
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
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
avg.sh best exp/conformer/checkpoints 20
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20
```
## Pretrained Model
You can get the pretrained transformer or conformer from [this](../../../docs/source/released_model.md).

using the `tar` scripts to unpack the model and then you can use the script to test the model.

For example:
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
tar xzvf asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
source path.sh
# If you have process the data and get the manifest file， you can skip the following 2 steps
bash local/data.sh --stage -1 --stop_stage -1
bash local/data.sh --stage 2 --stop_stage 2
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20
```
The performance of the released models are shown in [here](./RESULTS.md).

## Stage 4: CTC Alignment 
If you want to get the alignment between the audio and the text, you can use the ctc alignment. The code of this stage is shown below:
```bash
 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
     # ctc alignment of test data
     CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
 fi
```
If you want to train the model, test it and do the alignment, you can use the script below to execute stage 0, stage 1, stage 2, and stage 3 :
```bash
bash run.sh --stage 0 --stop_stage 4
```
or if you only need to train a model and do the alignment, you can use these scripts to escape stage 3(test stage):
```bash
bash run.sh --stage 0 --stop_stage 2
bash run.sh --stage 4 --stop_stage 4
```
or you can also use these scripts in the command line (only use CPU).
```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/conformer.yaml conformer
avg.sh best exp/conformer/checkpoints 20
# test stage is optional
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20
CUDA_VISIBLE_DEVICES= ./local/align.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20
```
## Stage 5: Single Audio File Inference
In some situations, you want to use the trained model to do the inference for the single audio file. You can use stage 5. The code is shown below
```bash
 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
     # test a single .wav file
     CUDA_VISIBLE_DEVICES=0 ./local/test_wav.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} ${audio_file} || exit -1
 fi
```
you can train the model by yourself using ```bash run.sh --stage 0 --stop_stage 3```, or you can download the pretrained model through the script below:
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
tar xzvf asr1_conformer_librispeech_ckpt_0.1.1.model.tar.gz
```
You can download the audio demo:
```bash
wget -nc https://paddlespeech.bj.bcebos.com/datasets/single_wav/en/demo_002_en.wav -P data/
```
You need to prepare an audio file or use the audio demo above, please confirm the sample rate of the audio is 16K. You can get the result of the audio demo by running the script below.
```bash
CUDA_VISIBLE_DEVICES= ./local/test_wav.sh conf/conformer.yaml conf/tuning/decode.yaml exp/conformer/checkpoints/avg_20 data/demo_002_en.wav
```
