bash prepare.sh ds2_params_whole_train_infer.txt whole_train_infer
cd ../../examples/aishell/s0
source path.sh
bash ../../../tests/chains/test.sh ../../../tests/chains/ds2_params_whole_train_infer.txt whole_train_infer
cd ../../../tests/chains
