#!/bin/bash

function abort(){
    echo "Your commit not fit PaddlePaddle code style" 1>&2
    echo "Please use pre-commit scripts to auto-format your code" 1>&2
    exit 1
}


trap 'abort' 0
set -e

source tools/venv/bin/activate

python3 --version

if ! pre-commit run -a ; then
  ls -lh
  git diff  --exit-code
  exit 1
fi

trap : 0
