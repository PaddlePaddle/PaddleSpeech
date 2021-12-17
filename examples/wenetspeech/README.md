* asr0 - deepspeech2 Streaming/Non-Streaming
* asr1 - transformer/conformer Streaming/Non-Streaming
* asr2 - transformer/conformer Streaming/Non-Streaming with Kaldi feature

# [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech)

A 10000+ Hours Multi-domain Chinese Corpus for Speech Recognition

## Description

### Creation

All the data are collected from YouTube and Podcast. Optical character recognition (OCR) and automatic speech recognition (ASR) techniques are adopted to label each YouTube and Podcast recording, respectively. To improve the quality of the corpus, we use a novel end-to-end label error detection method to further validate and filter the data.

### Categories

In summary, WenetSpeech groups all data into 3 categories, as the following table shows:

| Set        | Hours | Confidence  | Usage                                 |
|------------|-------|-------------|---------------------------------------|
| High Label | 10005 | >=0.95      | Supervised Training                   |
| Weak Label | 2478  | [0.6, 0.95] | Semi-supervised or noise training     |
| Unlabel    | 9952  | /           | Unsupervised training or Pre-training |
| In Total   | 22435 | /           | All above                             |

### High Label Data

We classify the high label into 10 groups according to its domain, speaking style, and scenarios.

| Domain      | Youtube | Podcast | Total  |
|-------------|---------|---------|--------|
| audiobook   | 0       | 250.9   | 250.9  |
| commentary  | 112.6   | 135.7   | 248.3  |
| documentary | 386.7   | 90.5    | 477.2  |
| drama       | 4338.2  | 0       | 4338.2 |
| interview   | 324.2   | 614     | 938.2  |
| news        | 0       | 868     | 868    |
| reading     | 0       | 1110.2  | 1110.2 |
| talk        | 204     | 90.7    | 294.7  |
| variety     | 603.3   | 224.5   | 827.8  |
| others      | 144     | 507.5   | 651.5  |
| Total       | 6113    | 3892    | 10005  |

As shown in the following table, we provide 3 training subsets, namely `S`, `M`, and `L` for building ASR systems on different data scales.

| Training Subsets | Confidence  | Hours |
|------------------|-------------|-------|
| L                | [0.95, 1.0] | 10005 |
| M                | 1.0         | 1000  |
| S                | 1.0         | 100   |

### Evaluation Sets

| Evaluation Sets | Hours | Source       | Description                                                                             |
|-----------------|-------|--------------|-----------------------------------------------------------------------------------------|
| DEV             | 20    | Internet     | Specially designed for some speech tools which require cross-validation set in training |
| TEST\_NET       | 23    | Internet     | Match test                                                                              |
| TEST\_MEETING   | 15    | Real meeting | Mismatch test which is a far-field, conversational, spontaneous, and meeting dataset   |
