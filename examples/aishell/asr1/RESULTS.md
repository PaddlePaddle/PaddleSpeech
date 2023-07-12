# Aishell

## RoFormer Streaming
paddle version: 2.5.0  
paddlespeech version: 1.5.0

Need set `decoding.decoding_chunk_size=16` when decoding.

> chunk_size=16, ((16 - 1) * 4 + 7) * 10ms = (16 * 4 + 3) * 10ms = 670ms

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| roformer | 44.80M | conf/chunk_roformer.yaml | spec_aug | test | attention | 16, -1 | - |  |  
| roformer | 44.80M | conf/chunk_roformer.yaml | spec_aug | test | ctc_greedy_search | 16, -1 | - |  |  
| roformer | 44.80M | conf/chunk_roformer.yaml | spec_aug | test | ctc_prefix_beam_search | 16, -1 | - |  |  
| roformer | 44.80M | conf/chunk_roformer.yaml | spec_aug | test | attention_rescoring | 16, -1 |  - |  |  

## Conformer
paddle version: 2.2.2  
paddlespeech version: 1.0.1
| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | attention | - | 0.0522 |
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | ctc_greedy_search | - | 0.0481 |
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | ctc_prefix_beam_search | - | 0.0480 | 
| conformer | 47.07M  | conf/conformer.yaml | spec_aug | test | attention_rescoring | - | 0.0460 | 


## Conformer Streaming
paddle version: 2.2.2  
paddlespeech version: 1.4.1  
Need set `decoding.decoding_chunk_size=16` when decoding.

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | attention | 16, -1 | - | 0.056102 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | ctc_greedy_search | 16, -1 | - | 0.058160 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | ctc_prefix_beam_search | 16, -1 | - | 0.058160 |  
| conformer | 47.06M | conf/chunk_conformer.yaml | spec_aug | test | attention_rescoring | 16, -1 |  - | 0.051968 |  


## Transformer 

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | attention | 3.8103787302970886 | 0.056588 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | ctc_greedy_search | 3.8103787302970886 | 0.059932 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | ctc_prefix_beam_search | 3.8103787302970886 | 0.059989 |  
| transformer | 31.95M  | conf/transformer.yaml | spec_aug | test | attention_rescoring | 3.8103787302970886 | 0.052273 |  
