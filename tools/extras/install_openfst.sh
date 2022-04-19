#!/bin/bash

set -e
set -x

# need support c++17, so need gcc >= 8
# openfst
openfst=openfst-1.8.1
shared=true
WGET="wget -c --no-check-certificate"

test -e ${openfst}.tar.gz || $WGET http://www.openfst.org/twiki/pub/FST/FstDownload/${openfst}.tar.gz
test -d ${openfst} || tar -xvf ${openfst}.tar.gz && chown -R root:root ${openfst}


if [ $shared == true ];then
    pushd ${openfst} && ./configure --enable-shared --enable-compact-fsts  --enable-compress   --enable-const-fsts   --enable-far    --enable-linear-fsts   --enable-lookahead-fsts  --enable-mpdt  --enable-ngram-fsts   --enable-pdt    --enable-python   --enable-special  --enable-bin  --enable-grm --prefix ${PWD}/install && popd
else
    pushd ${openfst} && ./configure --enable-static --enable-compact-fsts  --enable-compress   --enable-const-fsts   --enable-far    --enable-linear-fsts   --enable-lookahead-fsts  --enable-mpdt  --enable-ngram-fsts   --enable-pdt    --enable-python   --enable-special  --enable-bin  --enable-grm --prefix ${PWD}/install && popd
fi
pushd ${openfst} && make -j &&  make install && popd


suffix_path=$(python3 -c 'import sysconfig; import os; from pathlib import Path; site = sysconfig.get_paths()["purelib"]; site=Path(site); pwd = os.getcwd(); suffix = site.parts[-2:]; print(os.path.join(*suffix));')
wfst_so_path=${PWD}/${openfst}/install/lib/${suffix_path}
cp ${wfst_so_path}/pywrapfst.* $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
