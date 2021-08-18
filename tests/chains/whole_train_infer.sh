bash prepare.sh ds2_params_whole_train_infer.txt whole_train_infer
cd ../examples/aishell/s0
source path.sh
bash ../../../tests/test.sh ../../../tests/ds2_params_whole_train_infer.txt whole_train_infer
cd ../../../
