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

The `avg CER` of text normalization is: 0.006388318503308237
