# Aishell-1

## Deepspeech2 Streaming

| Model | Number of Params | Release | Config | Test set | Valid Loss | CER | 
| --- | --- | --- | --- | --- | --- | --- | 
| DeepSpeech2 | 45.18M | r0.2.0 | conf/deepspeech2_online.yaml + spec aug + fbank161 | test | 7.679287910461426 | 0.0718 |
| DeepSpeech2 | 45.18M | r0.2.0 | conf/deepspeech2_online.yaml + spec aug | test | 7.708217620849609| 0.078 |
| DeepSpeech2 | 45.18M | v2.2.0 | conf/deepspeech2_online.yaml + spec aug | test | 7.994938373565674 | 0.080 |  

## Deepspeech2 Non-Streaming

| Model | Number of Params | Release | Config | Test set | Valid Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- |  
| DeepSpeech2 | 58.4M | v2.2.0 | conf/deepspeech2.yaml + spec aug | test | 5.738585948944092 | 0.064000 |  
| DeepSpeech2 | 58.4M | v2.1.0 | conf/deepspeech2.yaml + spec aug | test | 7.483316898345947 | 0.077860 |  
| DeepSpeech2 | 58.4M | v2.1.0 | conf/deepspeech2.yaml | test | 7.299022197723389 | 0.078671 |
| DeepSpeech2 | 58.4M | v2.0.0 | conf/deepspeech2.yaml | test | - | 0.078977 |  
| --- | --- | --- | --- | --- | --- | --- |  
| DeepSpeech2 | 58.4M | v1.8.5 | - | test | - | 0.080447 |
