## Pack Model

pack model to tar.gz, e.g.

```bash
./utils/pack_model.sh  --preprocess_conf conf/preprocess.yaml --dict data/vocab.txt conf/conformer.yaml '' data/mean_std.json exp/conformer/checkpoints/wenetspeec
h.pdparams 

```

show model.tar.gz
```
tar tf model.tar.gz 
```
