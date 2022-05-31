# This contains the locations of binarys build required for running the examples.

SPEECHX_ROOT=$PWD/../../../
SPEECHX_BUILD=$SPEECHX_ROOT/build/speechx

SPEECHX_TOOLS=$SPEECHX_ROOT/tools
TOOLS_BIN=$SPEECHX_TOOLS/valgrind/install/bin

[ -d $SPEECHX_BUILD ] || { echo "Error: 'build/speechx' directory not found. please ensure that the project build successfully"; }

export LC_AL=C

SPEECHX_BIN=$SPEECHX_BUILD/protocol/websocket
export PATH=$PATH:$SPEECHX_BIN:$TOOLS_BIN
