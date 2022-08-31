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
The pretrained model can be downloaded here:

        [ernie_linear_p3_iwslt2012_zh_ckpt_0.1.1.zip](https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_iwslt2012_zh_ckpt_0.1.1.zip).

        [ernie-3.0-base.tar.gz](https://paddlespeech.bj.bcebos.com/punc_restore/ernie-3.0-base.tar.gz)

        [ernie-3.0-medium.tar.gz](https://paddlespeech.bj.bcebos.com/punc_restore/ernie-3.0-medium.tar.gz)

        [ernie-3.0-micro.tar.gz](https://paddlespeech.bj.bcebos.com/punc_restore/ernie-3.0-micro.tar.gz)

        [ernie-mini.tar.gz](https://paddlespeech.bj.bcebos.com/punc_restore/ernie-mini.tar.gz)

        [ernie-nano.tar.gz](https://paddlespeech.bj.bcebos.com/punc_restore/ernie-nano.tar.gz)

        [ernie-tiny.tar.gz](https://paddlespeech.bj.bcebos.com/punc_restore/ernie-tiny.tar.gz)

### Test Result
- Ernie 1.0
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.510955  |0.526462  |0.820755  |0.619391|
    |Recall     |0.517433  |0.564179  |0.861386  |0.647666|
    |F1         |0.514173  |0.544669  |0.840580  |0.633141|
- Ernie-tiny
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.733177  |0.721448  |0.754717  |0.736447|
    |Recall     |0.380740  |0.524646  |0.733945  |0.546443|
    |F1         |0.501204  |0.607506  |0.744186  |0.617632|
- Ernie-3.0-base-zh
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.805947  |0.764160  |0.858491  |0.809532|
    |Recall     |0.399070  |0.567978  |0.850467  |0.605838|
    |F1         |0.533817  |0.651623  |0.854460  |0.679967|
- Ernie-3.0-medium-zh
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.730829  |0.699164  |0.707547  |0.712514|
    |Recall     |0.388196  |0.533286  |0.797872  |0.573118|
    |F1         |0.507058  |0.605062  |0.750000  |0.620707|
- Ernie-3.0-mini-zh
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.757433  |0.708449  |0.707547  |0.724477|
    |Recall     |0.355752  |0.506977  |0.735294  |0.532674|
    |F1         |0.484121  |0.591015  |0.721154  |0.598763|
- Ernie-3.0-micro-zh
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.733959  |0.679666  |0.726415  |0.713347|
    |Recall     |0.332742  |0.483487  |0.712963  |0.509731|
    |F1         |0.457896  |0.565033  |0.719626  |0.580852|
- Ernie-3.0-nano-zh
    |       |COMMA  |  PERIOD | QUESTION | OVERALL|
    |:-----:|:-----:|:-----:|:-----:|:-----:|  
    |Precision  |0.693271  |0.682451  |0.754717  |0.710146|
    |Recall     |0.327784  |0.491968  |0.666667  |0.495473|
    |F1         |0.445114  |0.571762  |0.707965  |0.574947|
