# Text Normalization
For text normalization, the test data is  `data/textnorm_test_cases.txt`, we use `|` as the separator of raw_data and normed_data.

We use `CER` as an evaluation criterion.
## Start
Run the command below to get the results of the test.
```bash
cd ../../../tools
bash extras/install_sclite.sh
cd -
./run.sh
```
The `avg CER` of text normalization is: 0.00730093543235227
```text
      ,-----------------------------------------------------------------.
      |        | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      |--------+--------------+-----------------------------------------|
      | Sum/Avg|  125    2254 | 99.4    0.1    0.5    0.2    0.8    4.8 |
      `-----------------------------------------------------------------'
```
