# LibriSpeech

## Wav2VecASR
train: Epoch 1, 1*V100-32G, batchsize: 6

| Model | Params | Config | Augmentation| Test set | Decode method | WER |  
| --- | --- | --- | --- | --- | --- | --- |
| wav2vec2ASR | 302.86 M | conf/wav2vec2ASR.yaml | spec_aug | test-clean | greedy search | 0.019 |  
