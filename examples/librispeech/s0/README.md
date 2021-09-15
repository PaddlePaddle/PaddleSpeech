# LibriSpeech

## Data
| Data Subset | Duration in Seconds |
| --- | --- |
| data/manifest.train |  0.83s ~ 29.735s |
| data/manifest.dev | 1.065 ~ 35.155s |  
| data/manifest.test-clean | 1.285s ~ 34.955s |

## Deepspeech2

| Model | Params | release |  Config | Test set | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- |  
| DeepSpeech2 | 42.96M | 2.2.0 | conf/deepspeech2.yaml + spec_aug | test-clean | 14.49190807 | 0.067283 |  
| DeepSpeech2 | 42.96M | 2.1.0 | conf/deepspeech2.yaml | test-clean | 15.184467315673828 | 0.072154 |  
| DeepSpeech2 | 42.96M | 2.0.0 | conf/deepspeech2.yaml | test-clean | - | 0.073973 |  
| DeepSpeech2 | 42.96M | 1.8.5 | - | test-clean | - | 0.074939 |  
