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

                     SYSTEM SUMMARY PERCENTAGES by SPEAKER  

   ,------------------------------------------------------------------------.
   |                           ./exp/g2p/text.g2p                           |
   |------------------------------------------------------------------------|
   | SPKR | # Snt    # Wrd  |  Corr      Sub     Del    Ins    Err    S.Err |
   |------+-----------------+-----------------------------------------------|
   | bak  |  9996   299181  | 290969    8198      14     14   8226    5249  |
   |========================================================================|
   | Sum  |  9996   299181  | 290969    8198      14     14   8226    5249  |
   |========================================================================|
   | Mean |9996.0  299181.0 |290969.0  8198.0   14.0   14.0  8226.0  5249.0 |
   | S.D. |  0.0      0.0   |   0.0      0.0     0.0    0.0    0.0     0.0  |
   |Median|9996.0  299181.0 |290969.0  8198.0   14.0   14.0  8226.0  5249.0 |
   `------------------------------------------------------------------------'

                     SYSTEM SUMMARY PERCENTAGES by SPEAKER  

     ,--------------------------------------------------------------------.
     |                         ./exp/g2p/text.g2p                         |
     |--------------------------------------------------------------------|
     | SPKR   | # Snt    # Wrd  | Corr    Sub    Del    Ins    Err  S.Err |
     |--------+-----------------+-----------------------------------------|
     | bak    |  9996   299181  | 97.3    2.7    0.0    0.0    2.7   52.5 |
     |====================================================================|
     | Sum/Avg|  9996   299181  | 97.3    2.7    0.0    0.0    2.7   52.5 |
     |====================================================================|
     |  Mean  |9996.0  299181.0 | 97.3    2.7    0.0    0.0    2.7   52.5 |
     |  S.D.  |  0.0      0.0   |  0.0    0.0    0.0    0.0    0.0    0.0 |
     | Median |9996.0  299181.0 | 97.3    2.7    0.0    0.0    2.7   52.5 |
     `--------------------------------------------------------------------'
```

The `avg CER` of text normalization is: 0.006388318503308237
```text

                     SYSTEM SUMMARY PERCENTAGES by SPEAKER  

       ,----------------------------------------------------------------.
       |                     ./exp/textnorm/text.tn                     |
       |----------------------------------------------------------------|
       | SPKR | # Snt  # Wrd | Corr     Sub    Del    Ins    Err  S.Err |
       |------+--------------+------------------------------------------|
       | utt  |  125    2254 | 2241       2     11      2     15      4 |
       |================================================================|
       | Sum  |  125    2254 | 2241       2     11      2     15      4 |
       |================================================================|
       | Mean |125.0  2254.0 |2241.0    2.0   11.0    2.0   15.0    4.0 |
       | S.D. |  0.0    0.0  |  0.0     0.0    0.0    0.0    0.0    0.0 |
       |Median|125.0  2254.0 |2241.0    2.0   11.0    2.0   15.0    4.0 |
       `----------------------------------------------------------------'

                     SYSTEM SUMMARY PERCENTAGES by SPEAKER  

      ,-----------------------------------------------------------------.
      |                     ./exp/textnorm/text.tn                      |
      |-----------------------------------------------------------------|
      | SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      |--------+--------------+-----------------------------------------|
      | utt    |  125    2254 | 99.4    0.1    0.5    0.1    0.7    3.2 |
      |=================================================================|
      | Sum/Avg|  125    2254 | 99.4    0.1    0.5    0.1    0.7    3.2 |
      |=================================================================|
      |  Mean  |125.0  2254.0 | 99.4    0.1    0.5    0.1    0.7    3.2 |
      |  S.D.  |  0.0    0.0  |  0.0    0.0    0.0    0.0    0.0    0.0 |
      | Median |125.0  2254.0 | 99.4    0.1    0.5    0.1    0.7    3.2 |
      `-----------------------------------------------------------------'
```
