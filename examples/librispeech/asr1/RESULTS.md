# LibriSpeech

## Conformer
train: Epoch 70, 4 V100-32G, best avg: 20

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-clean | attention | 6.433612394332886 | 0.039771 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-clean | ctc_greedy_search | 6.433612394332886 | 0.040342 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-clean | ctc_prefix_beam_search | 6.433612394332886 | 0.040342 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-clean | attention_rescoring | 6.433612394332886 | 0.033761 |  


## Conformer Streaming

| Model | Params | Config | Augmentation| Test set | Decode method | Chunk Size & Left Chunks | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | attention | 16, -1 | 7.11 | 0.063193 |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | ctc_greedy_search | 16, -1 | 7.11 | 0.082394 |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | ctc_prefix_beam_search | 16, -1 | 7.11 | 0.082156 |  
| conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug + shift | test-clean | attention_rescoring | 16, -1 | 7.11 | 0.071000 |  


## Transformer

train: Epoch 120, 4 V100-32G, 27 Day, best avg: 10

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | WER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | attention | 6.382194232940674 | 0.049661 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | ctc_greedy_search | 6.382194232940674 | 0.049566 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | ctc_prefix_beam_search | 6.382194232940674 | 0.049585 |  
| transformer | 32.52 M | conf/transformer.yaml | spec_aug  | test-clean | attention_rescoring | 6.382194232940674 | 0.038135 |
