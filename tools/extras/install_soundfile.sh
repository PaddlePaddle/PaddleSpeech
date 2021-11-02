# install package libsndfile

WGET=wget --no-check-certificate

SOUNDFILE=libsndfile-1.0.28
SOUNDFILE_LIB=${SOUNDFILE}tar.gz

echo "Install package libsndfile into default system path."
test -e ${SOUNDFILE_LIB} || ${WGET} -c "http://www.mega-nerd.com/libsndfile/files/${SOUNDFILE_LIB}"
if [ $? != 0 ]; then
    echo "Download ${SOUNDFILE_LIB} failed !!!"
    exit 1
fi

tar -zxvf ${SOUNDFILE_LIB}
pushd ${SOUNDFILE}
./configure > /dev/null && make > /dev/null && make install > /dev/null
popd