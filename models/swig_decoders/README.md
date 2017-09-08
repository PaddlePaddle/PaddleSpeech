
The decoders for deployment developed in C++ are a better alternative for the prototype decoders in Pytthon, with more powerful performance in both speed and accuracy.

### Installation

The build depends on several open-sourced projects, first clone or download them to current directory (i.e., `deep_speech_2/deploy`)

- [**KenLM**](https://github.com/kpu/kenlm/): Faster and Smaller Language Model Queries

```shell
git clone https://github.com/kpu/kenlm.git
```

- [**OpenFst**](http://www.openfst.org/twiki/bin/view/FST/WebHome): A library for finite-state transducers

```shell
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
tar -xzvf openfst-1.6.3.tar.gz
```


- [**ThreadPool**](http://progsch.net/wordpress/): A library for C++ thread pool

```shell
git clone https://github.com/progschj/ThreadPool.git
```

- [**SWIG**](http://www.swig.org): A tool that provides the Python interface for the decoders, please make sure it being installed.

Then run the setup

```shell
python setup.py install --num_processes 4
cd ..
```

### Usage

The decoders for deployment share almost the same interface with the prototye decoders in Python. After the installation succeeds, these decoders are very convenient for call in Python, and a complete example in ```deploy.py``` can be refered.

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
