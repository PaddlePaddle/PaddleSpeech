# Aishell-1

## Data
| Data Subset | Duration in Seconds |
| data/manifest.train |  1.23 ~ 14.53125 |
| data/manifest.dev  | 1.645 ~ 12.533 |  
| data/manifest.test | 1.859125 ~ 14.6999375 |

## Deepspeech2

| Model | Params | Release | Config | Test set | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- |  
| DeepSpeech2 | 58.4M | 2.2.0 | conf/deepspeech2.yaml + spec aug | test | 6.016139030456543 | 0.066549 |  
| --- | --- | --- | --- | --- | --- | --- |  
| DeepSpeech2 | 58.4M | 7181e427 | conf/deepspeech2.yaml + spec aug | test | 5.71956205368042 | 0.064287 |  
| DeepSpeech2 | 58.4M | 2.1.0 | conf/deepspeech2.yaml + spec aug | test | 7.483316898345947 | 0.077860 |  
| DeepSpeech2 | 58.4M | 2.1.0 | conf/deepspeech2.yaml | test | 7.299022197723389 | 0.078671 |
| DeepSpeech2 | 58.4M | 2.0.0 | conf/deepspeech2.yaml | test | - | 0.078977 |  
| --- | --- | --- | --- | --- | --- | --- |  
| DeepSpeech2 | 58.4M | 1.8.5 | - | test | - | 0.080447 |  
