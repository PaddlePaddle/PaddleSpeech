# Aishell

## Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | attention | - | 0.059858 |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | ctc_greedy_search | - | 0.062311 |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | - | 0.062196 |  
| conformer | 47.07M  | conf/conformer.yaml | spec_aug + shift | test | attention_rescoring | - | 0.054694 |  


## Chunk Conformer
Need set `decoding.decoding_chunk_size=16` when decoding.

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | attention | 16, -1 | - | 0.061939 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | ctc_greedy_search | 16, -1 | - | 0.070806 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | 16, -1 | - | 0.070739 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug + shift | test | attention_rescoring | 16, -1 |  - | 0.059400 |  


## Transformer 

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | attention | 3.8103787302970886 | 0.056588 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | ctc_greedy_search | 3.8103787302970886 | 0.059932 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | ctc_prefix_beam_search | 3.8103787302970886 | 0.059989 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | attention_rescoring | 3.8103787302970886 | 0.052273 |  
