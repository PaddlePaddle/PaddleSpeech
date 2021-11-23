#!/bin/bash

# download pytorch weight
wget https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/snapshot.ep.98 --no-check-certificate

# convert pytorch weight to paddlepaddle
python local/convert_torch_to_paddle.py \
--torch_ckpt snapshot.ep.98 \
--paddle_ckpt paddle.98.pdparams

# Or you can download converted weights
# wget https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/paddle.98.pdparams --no-check-certificate

if [ $? -ne 0 ]; then
    echo "Failed in downloading and coverting!"
    exit 1
fi

exit 0