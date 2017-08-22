### Installation
The setup of the decoder for deployment depends on the source code of [kenlm](https://github.com/kpu/kenlm/) and [openfst](http://www.openfst.org/twiki/bin/view/FST/WebHome), first clone kenlm and download openfst to current directory (i.e., `deep_speech_2/deploy`)

```shell
git clone https://github.com/kpu/kenlm.git
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
tar -xzvf openfst-1.6.3.tar.gz
```

Then run the setup

```shell
python setup.py install
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
