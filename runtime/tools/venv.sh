#!/bin/bash
set -ex

PYTHON=python3.8
test -d venv || virtualenv -p ${PYTHON} venv
