#!/bin/bash

WGET="wget --no-check-certificate"

# SCTK official repo does not have version tags. Here's the mapping:
# # 2.4.9 = 659bc36; 2.4.10 = d914e1b; 2.4.11 = 20159b5.
SCTK_GITHASH=20159b5
SCTK_CXFLAGS="-w -march=native"
CFLAGS="CFLAGS=${SCTK_CXFLAGS}"
CXXFLAGS="CXXFLAGS=-std=c++11 ${SCTK_CXFLAGS}"

MAKE=make


${WGET} -nv -T 10 -t 3 -O sctk-${SCTK_GITHASH}.tar.gz  https://github.com/usnistgov/SCTK/archive/${SCTK_GITHASH}.tar.gz; 
tar zxvf sctk-${SCTK_GITHASH}.tar.gz
rm -rf sctk-${SCTK_GITHASH} sctk
mv SCTK-${SCTK_GITHASH}* sctk-${SCTK_GITHASH}
ln -s sctk-${SCTK_GITHASH} sctk
touch sctk-${SCTK_GITHASH}.tar.gz

rm -f sctk/.compiled
CFLAGS="${SCTK_CXFLAGS}" CXXFLAGS="-std=c++11 ${SCTK_CXFLAGS}" ${MAKE} -C sctk config
CFLAGS="${SCTK_CXFLAGS}" CXXFLAGS="-std=c++11 ${SCTK_CXFLAGS}"  ${MAKE} -C sctk all doc
${MAKE} -C sctk install
touch sctk/.compiled