# TIMIT




### Transformer
| Model | Params | Config | Decode method          | PER    |
| --- | --- | --- | --- | --- |
| transformer | 5.17M | conf/transformer.yaml | attention              | 0.5531 |
| transformer | 5.17M | conf/transformer.yaml | ctc_greedy_search      | 0.3922 |
| transformer | 5.17M | conf/transformer.yaml | ctc_prefix_beam_search | 0.3768 |