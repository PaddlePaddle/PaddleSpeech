# TIMIT

### Transformer
| Model | Params | Config | Decode method | Loss |  PER |
| --- | --- | --- | --- | --- | --- |
| transformer | 5.17M | conf/transformer.yaml | attention              | 46.41119385 | 0.396950 |
| transformer | 5.17M | conf/transformer.yaml | ctc_greedy_search      | 46.41119385 | 0.182259 |
| transformer | 5.17M | conf/transformer.yaml | ctc_prefix_beam_search | 46.41119385 | 0.188080 |
| transformer | 5.17M | conf/transformer.yaml | attention_rescore      | 46.41119385 | 0.184199 |
