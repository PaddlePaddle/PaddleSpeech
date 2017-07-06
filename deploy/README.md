### Installation
The setup of the decoder for deployment depends on the source code of [kenlm](https://github.com/kpu/kenlm/), first clone it to current directory (i.e., `deep_speech_2/deploy`)

```shell
git clone https://github.com/kpu/kenlm.git
```

Then run the setup

```shell
sh setup.sh
```

After the installation succeeds, go back to the parent directory

```
cd ..
```

### Deployment

For GPU deployment

```
CUDA_VISIBLE_DEVICES=0 python deploy.py
```

For CPU deployment

```
python deploy.py --use_gpu=False
```

More help for arguments

```
python deploy.py --help
```
