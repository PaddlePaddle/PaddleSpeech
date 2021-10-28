# Chinese Text Frontend Example
Here's an example for Chinese text frontend, including g2p and text normalization.
## G2P
For g2p, we use BZNSYP's phone label as the ground truth and we delete silence tokens in labels and predicted phones.

You should Download BZNSYP from it's [Official Website](https://test.data-baker.com/data/index/source) and extract it. Assume the path to the dataset is `~/datasets/BZNSYP`.

We use `WER` as evaluation criterion.
## Text Normalization
For text normalization, the test data is  `data/textnorm_test_cases.txt`, we use `|` as the separator of raw_data and normed_data.

We use `CER` as evaluation criterion.
## Start
If you want to use sclite to get more detail information of WER, you should run the command below to make sclite first.
```bash
./make_sclite.sh
```
Run the command below to get the results of test.
```bash
./run.sh
```
The `avg WER` of g2p is: 0.027495061517943988
```text
     ,--------------------------------------------------------------------.
     |        | # Snt    # Wrd  | Corr    Sub    Del    Ins    Err  S.Err |
     |--------+-----------------+-----------------------------------------|
     | Sum/Avg|  9996   299181  | 97.3    2.7    0.0    0.0    2.7   52.5 |
     `--------------------------------------------------------------------'
```

The `avg CER` of text normalization is: 0.006388318503308237
```text
      ,-----------------------------------------------------------------.
      |        | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      |--------+--------------+-----------------------------------------|
      | Sum/Avg|  125    2254 | 99.4    0.1    0.5    0.1    0.7    3.2 |
      `-----------------------------------------------------------------'
```
