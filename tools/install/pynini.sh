set -e
set -x

pynini=pynini-2.1.4

test -e ${pynini}.tar.gz || wget http://www.openfst.org/twiki/pub/GRM/PyniniDownload/${pynini}.tar.gz
test -d ${pynini} || tar -xvf ${pynini}.tar.gz && chown -R root:root ${pynini}

#wfst_so_path=$(python3 -c 'import sysconfig; import os; from pathlib import Path; site = sysconfig.get_paths()["purelib"]; site=Path(site); suffix = ("/usr/local/lib",) + site.parts[-2:]; print(os.path.join(*suffix));')

pushd ${pynini} &&  python setup.py install && popd

#cp ${wfst_so_path}/pywrapfst.* $(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
