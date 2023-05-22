# LibriSpeech

## WavLMASR
Fintuning on train-clean-100
train: Epoch 16, 4*A800-80G, batchsize: 16, accum_grad: 8

| Model | Params | Config | Augmentation| Test set | Decode method | WER |  
| --- | --- | --- | --- | --- | --- | --- |
| WavLMASR | 326.16M | conf/wavlmasr.yaml | spec_aug | test-clean | greedy search | 0.0561 |  
