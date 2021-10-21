#!/bin/bash

#install auto-log
echo "Install auto_log into default system path"
rm -rf AutoLog || true
test -d AutoLog || git clone https://github.com/LDOUBLEV/AutoLog
if [ $? != 0 ]; then
    error_msg "Download auto_log failed !!!"
    exit 1
fi

pushd AutoLog
pip3 install -r requirements.txt
python3 setup.py install
popd

rm -rf AutoLog || true
