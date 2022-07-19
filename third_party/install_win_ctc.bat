@echo off

cd ctc_decoders
if not exist kenlm (
    git clone https://github.com/Doubledongli/kenlm.git
    @echo.
)

if not exist openfst-1.6.3 (
    echo "Download and extract openfst ..."
    git clone https://gitee.com/koala999/openfst.git
    ren openfst openfst-1.6.3
    @echo.
)

if not exist ThreadPool (
    git clone https://github.com/progschj/ThreadPool.git
    @echo.
)
echo "Install decoders ..."
python setup.py install --num_processes 4