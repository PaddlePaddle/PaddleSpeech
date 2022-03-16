# This contains the locations of binarys build required for running the examples.

SPEECHX_ROOT=$PWD/../..
SPEECHX_EXAMPLES=$SPEECHX_ROOT/build/examples
SPEECHX_BIN=$SPEECHX_EXAMPLES/nnet:$SPEECHX_EXAMPLES/decoder:$SPEECHX_EXAMPLES/feat

[ -d $SPEECHX_EXAMPLES ] || { echo "Error: 'build/examples' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

export PATH=$PATH:$SPEECHX_BIN
