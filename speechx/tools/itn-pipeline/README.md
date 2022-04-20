# README
* The grammars is base on the open source project below.
https://github.com/wenet-e2e/wenet-text-processing/tree/main/grammars
the author of open source project is xingchensong, I add his project and name here, because this work spends his lots of time and energy. I have done it before, it is not a easy job to write those sophisticated grammars.

I made some modifications to adapt the grammars to the itn decoder.

* The itn_src is the itn decoder, which from: https://github.com/SmileGoat/workspace/tree/master/itn

1. install openfst & thrax in tools: if you have installed, skip, and add the path in cn/Makefile
    open tools: make

2. compile rule_fst:
    open cn: make all

3. test: compile the binary inverse_text_normalizer_main in src
  bash test_fst.sh

Author: Yang Zhou
