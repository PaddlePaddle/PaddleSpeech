#!/bin/bash

# install python dependencies
if [ -f 'requirements.txt' ]; then
    pip install -r requirements.txt
fi
if [ $? != 0 ]; then
    echo "Install python dependencies failed !!!"
    exit 1
fi

echo "Install all dependencies successfully."
