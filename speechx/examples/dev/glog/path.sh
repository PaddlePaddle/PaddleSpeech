# This contains the locations of binarys build required for running the examples.

SPEECHX_ROOT=$PWD/../../../

SPEECHX_TOOLS=$SPEECHX_ROOT/tools
TOOLS_BIN=$SPEECHX_TOOLS/valgrind/install/bin


SPEECHX_EXAMPLES=$SPEECHX_ROOT/build/examples
[ -d $SPEECHX_EXAMPLES ] || { echo "Error: 'build/examples' directory not found. please ensure that the project build successfully"; }

SPEECHX_BIN=$SPEECHX_EXAMPLES/dev/glog
export PATH=$PATH:$SPEECHX_BIN:$TOOLS_BIN

export LC_AL=C
