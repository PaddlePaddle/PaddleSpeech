# TALCS
2023.1.6, commit id: fa724285f3b799b97b4348ad3b1084afc0764f9b (conformer)
2025.8.11, commit id: 4f62ff05b7c9974d5642b26306ff3c7140c84312 (chunk_conformer)

## Conformer
train: Epoch 100, 3 V100-32G, best avg: 10

| Model | Params | Config | Augmentation| Test set | Decode method | Loss | MER |  
| --- | --- | --- | --- | --- | --- | --- | --- |
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | attention | 9.85091028213501 | 0.102786 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | ctc_greedy_search | 9.85091028213501 | 0.103538 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | ctc_prefix_beam_search | 9.85091028213501 | 0.103317 |  
| conformer | 47.63 M | conf/conformer.yaml | spec_aug | test-set | attention_rescoring | 9.85091028213501 | 0.084374 | 
| chunk_conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug | test-set | attention | 9.897139549255371 | 0.080488 |
| chunk_conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug | test-set | ctc_greedy_search | 9.897139549255371 | 0.093244 |
| chunk_conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug | test-set | ctc_prefix_beam_search | 9.897139549255371 | 0.093251 |
| chunk_conformer | 47.63 M | conf/chunk_conformer.yaml | spec_aug | test-set | attention_rescoring | 9.897139549255371 | 0.079193 | 
