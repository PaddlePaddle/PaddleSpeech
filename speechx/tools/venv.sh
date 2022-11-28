#!/bin/bash
set -ex

PYTHON=python3.7
test -d venv || virtualenv -p ${PYTHON} venv
