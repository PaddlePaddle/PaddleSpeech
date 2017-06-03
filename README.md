# Deep Speech 2 on PaddlePaddle

## Installation

Please replace `$PADDLE_INSTALL_DIR` with your own paddle installation directory.

```
pip install -r requirements.txt
export LD_LIBRARY_PATH=$PADDLE_INSTALL_DIR/Paddle/third_party/install/warpctc/lib:$LD_LIBRARY_PATH
```

For some machines, we also need to install libsndfile1. Details to be added.

## Usage

### Preparing Data

```
cd data
python librispeech.py
cd ..
```

More help for arguments:

```
python librispeech.py --help
```

### Traininig

For GPU Training:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --trainer_count 4
```

For CPU Training:

```
python train.py --trainer_count 8 --use_gpu False
```

More help for arguments:

```
python train.py --help
```

### Inferencing

```
python infer.py
```

More help for arguments:

```
python infer.py --help
```
