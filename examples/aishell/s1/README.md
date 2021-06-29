# Aishell

## Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | attention | - | 0.059858 |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | ctc_greedy_search | - | 0.062311 |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | - | 0.062196 |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | attention_rescoring | - | 0.054694 |  


## Chunk Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | attention | 16 | - | 0.061939 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | ctc_greedy_search | 16 | - | 0.070806 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | 16 | - | 0.070739 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | attention_rescoring | 16 |  - | 0.059400 |  


## Transformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | ---|  
| transformer | - | conf/transformer.yaml | spec_aug + shift | test | attention | - | - |  
