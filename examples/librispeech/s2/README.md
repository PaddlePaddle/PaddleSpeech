# LibriSpeech

## Transformer
| Model | Params | Config | Augmentation| Test Set | Decode Method | Loss | WER % |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 32.52 M | conf/transformer.yaml | spec_aug | test-clean | attention | 6.395054340362549 | 4.2 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug | test-clean | ctc_greedy_search | 6.395054340362549 | 5.0 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug | test-clean | ctc_prefix_beam_search | 6.395054340362549 |  |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug | test-clean | attention_rescore | 6.395054340362549 |  |  
