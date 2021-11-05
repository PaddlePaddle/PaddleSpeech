# Text Normalization
For text normalization, the test data is  `data/textnorm_test_cases.txt`, we use `|` as the separator of raw_data and normed_data.

We use `CER` as evaluation criterion.
## Start
Run the command below to get the results of test.
```bash
./run.sh
```
The `avg CER` of text normalization is: 0.006388318503308237
```text
      ,-----------------------------------------------------------------.
      |        | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      |--------+--------------+-----------------------------------------|
      | Sum/Avg|  125    2254 | 99.4    0.1    0.5    0.1    0.7    3.2 |
      `-----------------------------------------------------------------'
```
