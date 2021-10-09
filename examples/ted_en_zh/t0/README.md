
# TED En-Zh

## Dataset

| Data Subset | Duration in Seconds |
| --- | --- |
| data/manifest.train | 0.942 ~ 60   |
| data/manifest.dev   | 1.151 ~ 39   |
| data/manifest.test  | 1.1 ~ 42.746 |

## Transformer
| Model | Params | Config | Char-BLEU |
| --- | --- | --- | --- |
| Transformer+ASR MTL | 50.26M | conf/transformer_joint_noam.yaml | 17.38 |
