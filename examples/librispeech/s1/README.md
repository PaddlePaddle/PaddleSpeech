# LibriSpeech

## Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-all | attention | 6.35 | 0.057117 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | attention | 6.35 | 0.030162 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | ctc_greedy_search | 6.35 | 0.037910 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | ctc_prefix_beam_search | 6.35 | 0.037761 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | attention_rescoring | 6.35 | 0.032115 |  

## Transformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 32.52 M | conf/transformer.yaml | spec_aug + shift | test-all | attention | 6.98 | 0.066500 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug + shift | test-clean | attention | 6.98 | 0.036 |  
