# This contains the locations of binarys build required for running the examples.

unset GREP_OPTIONS

ENGINE_ROOT=$PWD/../../../
ENGINE_BUILD=$ENGINE_ROOT/build/engine/asr

ENGINE_TOOLS=$ENGINE_ROOT/tools
TOOLS_BIN=$ENGINE_TOOLS/valgrind/install/bin

[ -d $ENGINE_BUILD ] || { echo "Error: 'build/runtime' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

export PATH=$PATH:$TOOLS_BIN:$ENGINE_BUILD/nnet:$ENGINE_BUILD/decoder:$ENGINE_BUILD/../common/frontend/audio:$ENGINE_BUILD/recognizer

#PADDLE_LIB_PATH=$(python -c "import os; import paddle; include_dir=paddle.sysconfig.get_include(); paddle_dir=os.path.split(include_dir)[0]; libs_dir=os.path.join(paddle_dir, 'libs'); fluid_dir=os.path.join(paddle_dir, 'fluid'); out=':'.join([libs_dir, fluid_dir]); print(out);")
export LD_LIBRARY_PATH=$PADDLE_LIB_PATH:$LD_LIBRARY_PATH
