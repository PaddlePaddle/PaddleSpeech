# LibriSpeech

## Conformer
| Model | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- |
| conformer | conf/conformer.yaml | spec_aug + shift | test-all | attention | test-all 6.35 | 0.057117 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test-clean | attention | test-all 6.35 | 0.030162 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test-clean | ctc_greedy_search | test-all 6.35 | 0.037910 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test-clean | ctc_prefix_beam_search | test-all 6.35 | 0.037761 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test-clean | attention_rescoring | test-all 6.35 | 0.032115 |  

## Transformer

| Model | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- |
| transformer | conf/transformer.yaml | spec_aug + shift | test-all | attention | test-all 6.98 | 0.066500 |  
| transformer | conf/transformer.yaml | spec_aug + shift | test-clean | attention | test-all 6.98 | 0.036 |  
