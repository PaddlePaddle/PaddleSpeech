# Aishell

## Conformer
| Model | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- |  
| conformer | conf/conformer.yaml | spec_aug + shift | test | attention | - | 0.059858 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test | ctc_greedy_search | - | 0.062311 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test | ctc_prefix_beam_search | - | 0.062196 |  
| conformer | conf/conformer.yaml | spec_aug + shift | test | attention_rescoring | - | 0.054694 |  

## Transformer

| Model | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | ---|  
| transformer | conf/transformer.yaml | spec_aug + shift | test | attention | - | - |  
