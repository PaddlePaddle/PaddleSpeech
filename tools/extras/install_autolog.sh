#!/bin/bash

#install auto-log
python -c "import auto_log"
if [ $? != 0 ]; then
    info_msg "Install auto_log into default system path"
    test -d AutoLog || git clone https://github.com/LDOUBLEV/AutoLog
    if [ $? != 0 ]; then
        error_msg "Download auto_log failed !!!"
        exit 1
    fi
    cd AutoLog
    pip install -r requirements.txt
    python setup.py install
    cd ..
    rm -rf AutoLog
fi