# Rhythm Prediction with CSMSC and AiShell3

## Get Started
### Data Preprocessing
```bash
./run.sh --stage 0 --stop-stage 0
```
### Model Training
```bash
./run.sh --stage 1 --stop-stage 1
```
### Testing
```bash
./run.sh --stage 2 --stop-stage 2
```
### Punctuation Restoration
```bash
./run.sh --stage 3 --stop-stage 3
```
## Pretrained Model
The pretrained model can be downloaded here:

[ernie-1.0_aishellcsmsc_ckpt_1.3.0.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/rhy_predict/ernie-1.0_aishellcsmsc_ckpt_1.3.0.zip)

And you should put it into `exp/YOUREXP/checkpoints` folder.

## Rhythm mapping
Four punctuation marks are used to denote the rhythm marks respectively:
|ryh_token|csmsc|aishll3|
|:---: |:---: |:---: |
|%|#1|%|
|`|#2||
|~|#3||
|$|#4|$|

## Prediction Results
|       |  #1  |  #2 |  #3  |  #4  |
|:-----:|:-----:|:-----:|:-----:|:-----:|  
|Precision  |0.90  |0.66  |0.91  |0.90|
|Recall     |0.92  |0.62  |0.83  |0.85|
|F1         |0.91  |0.64  |0.87  |0.87|
