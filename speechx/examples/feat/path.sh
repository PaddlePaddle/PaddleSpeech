# This contains the locations of binarys build required for running the examples.

SPEECHX_ROOT=$PWD/../..
SPEECHX_EXAMPLES=$SPEECHX_ROOT/build/examples

SPEECHX_TOOLS=$SPEECHX_ROOT/tools
TOOLS_BIN=$SPEECHX_TOOLS/valgrind/install/bin

[ -d $SPEECHX_EXAMPLES ] || { echo "Error: 'build/examples' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

SPEECHX_BIN=$SPEECHX_EXAMPLES/feat
export PATH=$PATH:$SPEECHX_BIN:$TOOLS_BIN
