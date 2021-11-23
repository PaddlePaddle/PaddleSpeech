# WenetSpeech


## Conformer

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | dev | attention |  |  |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | test net | ctc_greedy_search |  |  |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | test meeting | ctc_prefix_beam_search |  |  |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | test net | attention_rescoring |  |  |  



## Conformer Pretrain Model

Pretrain model from http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | attention | - | 0.048456 |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | ctc_greedy_search | - | 0.052534 |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | ctc_prefix_beam_search | - | 0.052915 |  
| conformer | 32.52 M | conf/conformer.yaml | spec_aug  | aishell1 | attention_rescoring | - | 0.047904 |  
