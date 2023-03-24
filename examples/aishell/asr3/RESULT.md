# AISHELL

## Version

* paddle version: develop (commit id: daea892c67e85da91906864de40ce9f6f1b893ae)
* paddlespeech version: develop (commit id: c14b4238b256693281e59605abff7c9435b3e2b2)
* paddlenlp version: 2.5.2 

## Device
* python: 3.7
* cuda: 10.2
* cudnn: 7.6

## Result
train: Epoch 80, 2*V100-32G, batchsize:5
| Model | Params | Config | Augmentation| Test set | Decode method | WER |  
| --- | --- | --- | --- | --- | --- | --- |
| wav2vec2ASR | 324.49 M | conf/wav2vec2ASR.yaml | spec_aug | test-set | greedy search | 5.1009 |  
