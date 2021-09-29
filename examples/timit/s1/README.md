# TIMIT

### Transformer
| Model | Params | Config | Decode method | Loss |  PER |
| --- | --- | --- | --- | --- |
| transformer | 5.17M | conf/transformer.yaml | attention              | 49.25688171386719 | 0.510742 |
| transformer | 5.17M | conf/transformer.yaml | ctc_greedy_search      | 49.25688171386719 | 0.382398 |
| transformer | 5.17M | conf/transformer.yaml | ctc_prefix_beam_search | 49.25688171386719 | 0.367429 |
| transformer | 5.17M | conf/transformer.yaml | attention_rescore      | 49.25688171386719 | 0.357173 |  
