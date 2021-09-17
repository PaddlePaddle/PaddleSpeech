bash prepare.sh ds2_params_lite_train_infer.txt lite_train_infer
cd ../../examples/tiny/s0
source path.sh
bash ../../../tests/chains/test.sh ../../../tests/chains/ds2_params_lite_train_infer.txt lite_train_infer
cd ../../../tests/chains
