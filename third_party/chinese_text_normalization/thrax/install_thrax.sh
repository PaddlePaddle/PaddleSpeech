#!/bin/bash
## This script should be placed under $KALDI_ROOT/tools/extras/, and see INSTALL.txt for installation guide
if [ ! -f thrax-1.2.9.tar.gz ]; then
    wget http://www.openfst.org/twiki/pub/GRM/ThraxDownload/thrax-1.2.9.tar.gz
    tar -zxf thrax-1.2.9.tar.gz
fi
cd thrax-1.2.9
OPENFSTPREFIX=`pwd`/../openfst
LDFLAGS="-L${OPENFSTPREFIX}/lib" CXXFLAGS="-I${OPENFSTPREFIX}/include" ./configure --prefix ${OPENFSTPREFIX}
make -j 10; make install
cd ..

