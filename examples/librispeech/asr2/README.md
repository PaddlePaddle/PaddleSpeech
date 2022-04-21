# Transformer/Conformer ASR with Librispeech ASR2

This example contains code used to train a Transformer or [Conformer](http://arxiv.org/abs/2008.03802) model with [Librispeech dataset](http://www.openslr.org/resources/12) and use some functions in kaldi.

To use this example, you need to install Kaldi first.

## Overview

All the scripts you need are in ```run.sh```. There are several stages in ```run.sh```, and each stage has its function.

| Stage | Function                                                     |
|:---- |:----------------------------------------------------------- |
| 0     | Process data. It includes: <br>       (1) Download the dataset <br>       (2) Calculate the CMVN of the train dataset <br>       (3) Get the vocabulary file <br>       (4) Get the manifest files of the train, development and test dataset<br>       (5) Get the sentencepiece model |
| 1     | Train the model                                              |
| 2     | Get the final model by averaging the top-k models, set k = 1 means to choose the best model |
| 3     | Test the final model performance                             |
| 4     | Join ctc decoder and use transformer language model to score |
| 5     | Get ctc alignment of test data using the final model         |
| 6     | Calculate the perplexity of transformer language model        |


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
`stage` denotes the number of the stage you want to start from in the experiments.
`stop stage` denotes the number of the stage you want to end at in the experiments. 
`conf_path` denotes the config path of the model.
`dict_path` denotes the path of the vocabulary file.
`avg_num` denotes the number K of top-K models you want to average to get the final model.
`ckpt` denotes the checkpoint prefix of the model, e.g. "transformer"

You can set the local variables (except `ckpt`) when you use `run.sh`

For example, you can set the `gpus` and `avg_num` when you use the command line.:
```bash
bash run.sh --gpus 0,1 --avg_num 10
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
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
```

After processing the data, the ``data`` directory will look like this:

```bash
data/
├── dev
├── dev_clean
├── dev-clean.meta
├── dev_org
├── dev_other
├── dev-other.meta
├── lang_char
├── manifest.dev
├── manifest.dev-clean
├── manifest.dev-clean.raw
├── manifest.dev-other
├── manifest.dev-other.raw
├── manifest.dev.raw
├── manifest.test-clean
├── manifest.test-clean.raw
├── manifest.test-other
├── manifest.test-other.raw
├── manifest.test.raw
├── manifest.train
├── manifest.train-clean-100.raw
├── manifest.train-clean-360.raw
├── manifest.train-other-500.raw
├── manifest.train.raw
├── temp1
├── temp2
├── temp3
├── test_clean
├── test-clean.meta
├── test_other
├── test-other.meta
├── train_960
├── train_960_org
├── train_clean_100
├── train-clean-100.meta
├── train_clean_360
├── train-clean-360.meta
├── train_other_500
├── train-other-500.meta
├── train_sp
└── train_sp_org
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
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
```
## Stage 2: Top-k Models Averaging
After training the model, we need to get the final model for testing and inference. In every epoch, the model checkpoint is saved, so we can choose the last K models and average the parameters of the models to get the final model. We can use stage 2 to do this, and the code is shown below:
```bash
 if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     # avg n best model
     avg.sh lastest exp/${ckpt}/checkpoints ${avg_num}
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
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
avg.sh best exp/transformer/checkpoints 10
```
## Stage 3: Model Testing
Stage 3 is to evaluate the model performance with an attention rescore decoder. The code of this stage is shown below:
```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # attetion resocre decoder
    ./local/test.sh ${conf_path} ${dict_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
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
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
avg.sh latest exp/transformer/checkpoints 10
CUDA_VISIBLE_DEVICES= ./local/test.sh conf/transformer.yaml data/train_960_unigram5000_units.txt exp/transformer/checkpoints/avg_10
```
## Stage 4: Model Testing with Join CTC Decoder
Stage 4 is to evaluate the model performance with the join ctc decoder. The code of this stage is shown below:
```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # join ctc decoder, use transformerlm to score
    ./local/recog.sh  --ckpt_prefix exp/${ckpt}/checkpoints/${avg_ckpt}
fi
```
If you want to train a model and test it, you can use the script below to execute stage 0, stage 1, stage 2, and stage 4 :
```bash
bash run.sh --stage 0 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
```
or you can run these scripts in the command line (only use CPU).
```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
avg.sh latest exp/transformer/checkpoints 10
./local/recog.sh  --ckpt_prefix exp/transformer/checkpoints/avg_10
```
## Pretrained Model
You can get the pretrained models from [this](../../../docs/source/released_model.md).

using the `tar` scripts to unpack the model and then you can use the script to test the model.

For example:
```bash
wget https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr2/asr2_transformer_librispeech_ckpt_0.1.1.model.tar.gz
tar xzvf asr2_transformer_librispeech_ckpt_0.1.1.model.tar.gz
source path.sh
# If you have process the data and get the manifest file， you can skip the following 2 steps
bash local/data.sh --stage -1 --stop_stage -1
bash local/data.sh --stage 2 --stop_stage 2

CUDA_VISIBLE_DEVICES= ./local/test.sh conf/transformer.yaml exp/ctc/checkpoints/avg_10
```
The performance of the released models are shown [here](./RESULTS.md).

Compare with [ESPNET](https://github.com/espnet/espnet/blob/master/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--transformer-lm-4-gpus) we using 8gpu, but the model size (aheads4-adim256) small than it.
## Stage 5: CTC Alignment 
If you want to get the alignment between the audio and the text, you can use the ctc alignment. The code of this stage is shown below:
```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # ctc alignment of test data
    CUDA_VISIBLE_DEVICES=0 ./local/align.sh ${conf_path} ${dict_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
```
If you want to train the model, test it and do the alignment, you can use the script below to execute stage 0, stage 1, stage 2, stage 3, stage 4, and stage 5:
```bash
bash run.sh --stage 0 --stop_stage 5
```
or if you only need to train a model and do the alignment, you can use these scripts to escape stage 3(test stage):
```bash
bash run.sh --stage 0 --stop_stage 2
bash run.sh --stage 5 --stop_stage 5
```
or you can also use these scripts in the command line (only use CPU).
```bash
. ./path.sh
. ./cmd.sh
bash ./local/data.sh
CUDA_VISIBLE_DEVICES= ./local/train.sh conf/transformer.yaml transformer
avg.sh best exp/transformer/checkpoints 20
CUDA_VISIBLE_DEVICES= ./local/align.sh conf/transformer.yaml data/train_960_unigram5000_units.txt exp/transformer/checkpoints/avg_10
```
## Stage 6: Perplexity Calculation 
This stage is for calculating the perplexity of the transformer language model. The code of this stage is shown below:
```bash
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ./local/cacu_perplexity.sh || exit -1
fi
```
If you only want to calculate the perplexity of the transformer language model, you can use this script:

```bash
bash run.sh --stage 6 --stop_stage 6
```
