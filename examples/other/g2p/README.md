# G2P
For g2p, we use BZNSYP's phone label as the ground truth and we delete silence tokens in labels and predicted phones.

You should Download BZNSYP from its [Official Website](https://test.data-baker.com/data/index/source) and extract it. Assume the path to the dataset is `~/datasets/BZNSYP`.

We use `WER` as an evaluation criterion.

# Start
Run the command below to get the results of the test.

```bash
./run.sh
```

The `avg WER` of g2p is: 0.024075726733983775

```text
     ,--------------------------------------------------------------------.
     |                         ./exp/g2p/text.g2p                         |
     |--------------------------------------------------------------------|
     | SPKR   | # Snt    # Wrd  | Corr    Sub    Del    Ins    Err  S.Err |
     | Sum/Avg|  9996   299181  | 97.6    2.4    0.0    0.0    2.4   49.0 |
     `--------------------------------------------------------------------'
```
