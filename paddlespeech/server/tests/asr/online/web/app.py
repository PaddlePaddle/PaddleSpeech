#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: zhendong.peng@mobvoi.com (Zhendong Peng)

import argparse

from flask import Flask, render_template

parser = argparse.ArgumentParser(description='training your network')
parser.add_argument('--port', default=19999, type=int, help='port id')
args = parser.parse_args()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=True)
