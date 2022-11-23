bash prepare.sh ds2_params_lite_train_infer.txt lite_train_infer
cd ../../../examples/tiny/asr0
source path.sh
bash ../../../tests/chains/ds2/test.sh ../../../tests/chains/ds2/ds2_params_lite_train_infer.txt lite_train_infer
cd ../../../tests/chains
