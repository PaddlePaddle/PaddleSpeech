bash prepare.sh ds2_params_lite_train_infer.txt lite_train_infer
cd ../examples/tiny/s0
source path.sh
bash ../../../tests/test.sh ../../../tests/ds2_params_lite_train_infer.txt lite_train_infer
cd ../../../
