set -e
set -x

# need support c++17, so need gcc >= 8
# openfst
openfst=openfst-1.8.1
shared=true

test -e ${openfst}.tar.gz || wget http://www.openfst.org/twiki/pub/FST/FstDownload/${openfst}.tar.gz
test -d ${openfst} || tar -xvf ${openfst}.tar.gz && chown -R root:root ${openfst}

wfst_so_path=$(python3 -c 'import sysconfig; import os; from pathlib import Path; site = sysconfig.get_paths()["purelib"]; site=Path(site); suffix = ("/usr/local/lib",) + site.parts[-2:]; print(os.path.join(*suffix));')

if [ $shared == true ];then
    pushd ${openfst} && ./configure --enable-shared --enable-compact-fsts  --enable-compress   --enable-const-fsts   --enable-far    --enable-linear-fsts   --enable-lookahead-fsts  --enable-mpdt  --enable-ngram-fsts   --enable-pdt    --enable-python   --enable-special  --enable-bin  --enable-grm --prefix ${PWD}/output && popd
else
    pushd ${openfst} && ./configure --enable-static --enable-compact-fsts  --enable-compress   --enable-const-fsts   --enable-far    --enable-linear-fsts   --enable-lookahead-fsts  --enable-mpdt  --enable-ngram-fsts   --enable-pdt    --enable-python   --enable-special  --enable-bin  --enable-grm --prefix ${PWD}/output && popd
fi
pushd ${openfst} && make -j &&  make install && popd

cp ${wfst_so_path}/pywrapfst.* $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
