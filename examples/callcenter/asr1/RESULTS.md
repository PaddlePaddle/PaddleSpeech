# MandarinK8

## Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 45.73 M | conf/conformer.yaml | spec_aug + shift | test | attention | 2.1794936656951904 | 0.102304 |  
| conformer | 45.73 M | conf/conformer.yaml | spec_aug + shift | test | ctc_greedy_search | 2.1794936656951904 | 0.084295 |  
| conformer | 45.73 M | conf/conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | 2.1794936656951904 | 0.084340 |  
| conformer | 45.73 M | conf/conformer.yaml | spec_aug + shift | test | attention_rescoring | 2.1794936656951904 | 0.081675 |  


## Chunk Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 45.73 M | conf/chunk_conformer.yaml | spec_aug + shift | test | attention | 16, -1 | 2.23287845  | 0.087982 |  
| conformer | 45.73 M | conf/chunk_conformer.yaml | spec_aug + shift | test | ctc_greedy_search | 16, -1 | 2.23287845  | 0.086962 |  
| conformer | 45.73 M | conf/chunk_conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | 16, -1 | 2.23287845 | 0.086741 |  
| conformer | 45.73 M | conf/chunk_conformer.yaml | spec_aug + shift | test | attention_rescoring | 16, -1 | 2.23287845 | 0.083495 |
