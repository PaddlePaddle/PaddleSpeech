# TALCS
2023.1.6, commit id: fa724285f3b799b97b4348ad3b1084afc0764f9b

## Conformer
train: Epoch 100, 3 V100-32G, best avg: 10

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | MER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | attention | 9.85091028213501 | 0.102786 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | ctc_greedy_search | 9.85091028213501 | 0.103538 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | ctc_prefix_beam_search | 9.85091028213501 | 0.103317 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | attention_rescoring | 9.85091028213501 | 0.084374 |  
