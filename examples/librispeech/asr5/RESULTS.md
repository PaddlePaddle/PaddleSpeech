# LibriSpeech

## hubertASR
Fintuning on train-clean-100
train: Epoch 3, 1*V100-32G, batchsize: 4, accum_grad: 8

| Model | Params | Config | Augmentation| Test set | Decode method | WER |  
| --- | --- | --- | --- | --- | --- | --- |
| hubertASR | 326.16M | conf/hubertASR.yaml | spec_aug | test-clean | greedy search | 0.05868 |  
