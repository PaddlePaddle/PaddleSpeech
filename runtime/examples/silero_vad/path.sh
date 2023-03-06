# This contains the locations of binarys build required for running the examples.

unset GREP_OPTIONS

ENGINE_ROOT=$PWD/../../
ENGINE_BUILD=$ENGINE_ROOT/build/engine/vad

ENGINE_TOOLS=$ENGINE_ROOT/tools
TOOLS_BIN=$ENGINE_TOOLS/valgrind/install/bin

[ -d $ENGINE_BUILD ] || { echo "Error: 'build/runtime' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

export PATH=$PATH:$TOOLS_BIN:$ENGINE_BUILD

export LD_LIBRARY_PATH=$PADDLE_LIB_PATH:$LD_LIBRARY_PATH
