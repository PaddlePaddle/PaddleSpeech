# LibriSpeech

## Conformer
| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | attention | 6.738649845123291 | 0.041159 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | ctc_greedy_search | 6.738649845123291 | 0.039847 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | ctc_prefix_beam_search | 6.738649845123291 | 0.039790 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug + shift | test-clean | attention_rescoring | 6.738649845123291 | 0.034617 |  


## Chunk Conformer
| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | attention | 16, -1 | 7.11 | 0.063193 |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | ctc_greedy_search | 16, -1 | 7.11 | 0.082394 |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | ctc_prefix_beam_search | 16, -1 | 7.11 | 0.082156 |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | attention_rescoring | 16, -1 | 7.11 | 0.071000 |  


## Transformer
| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | attention | 6.733129533131917 | 0.047874 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | ctc_greedy_search | 6.733129533131917 | 0.053922 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | ctc_prefix_beam_search | 6.733129533131917 | 0.053427 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | attention_rescoring | 6.733129533131917 | 0.041369 |  
