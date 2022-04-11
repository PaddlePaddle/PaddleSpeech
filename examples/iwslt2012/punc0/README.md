# Punctuation Restoration with IWLST2012-Zh

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
The pretrained model can be downloaded here [ernie_linear_p3_iwslt2012_zh_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_iwslt2012_zh_ckpt_0.1.1.zip).

### Test Result
- Ernie
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.510955  |0.526462  |0.820755  |0.619391|
    |Recall     |0.517433  |0.564179  |0.861386  |0.647666|
    |F1         |0.514173  |0.544669  |0.840580  |0.633141|
