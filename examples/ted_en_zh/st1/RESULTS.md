
# TED En-Zh

## Dataset

| Data Subset | Duration in Frames |
| --- | --- |
| data/manifest.train | 94.2 ~ 6000   |
| data/manifest.dev   | 115.1 ~ 3900   |
| data/manifest.test  | 110 ~ 4274.6 |

## Transformer
| Model | Params | Config | Val loss | Char-BLEU |
| --- | --- | --- | --- | --- |
| FAT + Transformer+ASR MTL | 50.26M | conf/transformer_mtl_noam.yaml | 69.91 | 20.26 |
| FAT + Transformer+ASR MTL with word reward | 50.26M | conf/transformer_mtl_noam.yaml | 62.86 | 20.80 |
