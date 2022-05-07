# WenetSpeech

## Conformer online

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
