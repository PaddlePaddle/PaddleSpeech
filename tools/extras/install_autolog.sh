#!/bin/bash

#install auto-log
echo "Install auto_log into default system path"
test -d AutoLog || git clone https://github.com/LDOUBLEV/AutoLog
if [ $? != 0 ]; then
    error_msg "Download auto_log failed !!!"
    exit 1
fi

pushd AutoLog
pip install -r requirements.txt
python setup.py install
popd

rm -rf AutoLog
