# TIMIT

### Transformer
| Model | Params | Config | Decode method | Loss |  PER |
| --- | --- | --- | --- | --- | --- |
| transformer | 5.17M | conf/transformer.yaml | attention              | 46.41119385 | 0.469993 |
| transformer | 5.17M | conf/transformer.yaml | ctc_greedy_search      | 46.41119385 | 0.297713 |
| transformer | 5.17M | conf/transformer.yaml | ctc_prefix_beam_search | 46.41119385 | 0.293555 |
| transformer | 5.17M | conf/transformer.yaml | attention_rescore      | 46.41119385 | 0.281081 |
