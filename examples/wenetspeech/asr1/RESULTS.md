# WenetSpeech

## Conformer Streaming

| Model | Params | Config | Augmentation| Test set | Decode method | Valid Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test net | attention | 9.329 | 0.1102 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test net | ctc_greedy_search | 9.329 | 0.1207 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test net | ctc_prefix_beam_search | 9.329 | 0.1203 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test net | attention_rescoring | 9.329  | 0.1100 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test meeting | attention | 9.329 | 0.1992 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test meeting | ctc_greedy_search | 9.329 | 0.1960 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test meeting | ctc_prefix_beam_search | 9.329 | 0.1946 |  
| conformer_online | 123.47 M | conf/chunk_conformer.yaml | spec_aug  | test meeting | attention_rescoring | 9.329  | 0.1879|  

## Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | dev | attention |  |  |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | test net | ctc_greedy_search |  |  |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | test meeting | ctc_prefix_beam_search |  |  |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | test net | attention_rescoring |  |  |  



## Conformer Pretrain Model

Pretrain model from http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | attention | - | 0.048456 |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | - | 0.052534 |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | - | 0.052915 |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | attention_rescoring | - | 0.047904 |  


## Conformer Steaming Pretrained Model

Pretrain model from https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention | 16 | 0.056273 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | 16 | 0.078918 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | 16 | 0.079080 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention_rescoring | 16 | 0.054401 |

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention | -1 | 0.050767 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | -1 | 0.061884 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | -1 | 0.062056 |  
| conformer | 32.52 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention_rescoring | -1 |  0.052110 |


## U2PP Steaming Pretrained Model

Pretrain model from https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_u2pp_wenetspeech_ckpt_1.3.0.model.tar.gz

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention | 16 | 0.057031 |  
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | 16 | 0.068826 |  
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | 16 | 0.069111 |  
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention_rescoring | 16 | 0.059213 |

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size | CER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention | -1 | 0.049256 |  
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | -1 | 0.052086 |  
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | -1 | 0.052267 |  
| conformer | 122.88 M | conf/chunk_conformer.yaml | spec_aug  | aishell1 | attention_rescoring | -1 |  0.047198 |
