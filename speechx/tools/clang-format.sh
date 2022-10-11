#!/bin/bash

find speechx -name '*.c' -o -name '*.h' -not -path "*kaldi*" | xargs -I{} clang-format -i  {}
