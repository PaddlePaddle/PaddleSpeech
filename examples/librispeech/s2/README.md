# LibriSpeech

| Model | Params | Config | Augmentation| Loss |
| --- | --- | --- | --- |
| transformer | 32.52 M | conf/transformer.yaml | spec_aug | 6.3197922706604 |


| Test Set | Decode Method | #Snt | #Wrd | Corr | Sub | Del | Ins | Err | S.Err |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test-clean | attention | 2620 | 52576 | 96.4 | 2.5 | 1.1 | 0.4 | 4.0 | 34.7 |  
| test-clean | ctc_greedy_search | 2620 | 52576 | 95.9 | 3.7 | 0.4 | 0.5 | 4.6 | 48.0 |  
| test-clean | ctc_prefix_beamsearch | 2620 | 52576 | 95.9 | 3.7 | 0.4 | 0.5 | 4.6 | 47.6 |  
| test-clean | attention_rescore | 2620 | 52576 | 96.8 | 2.9 | 0.3 | 0.4 | 3.7 | 38.0 |  
| test-clean | join_ctc_w/o_lm | 2620 | 52576 | 97.2 | 2.6 | 0.3 | 0.4 | 3.2 | 34.9 |  
