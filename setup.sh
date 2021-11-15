# Install conda dependencies
conda install -c conda-forge sox libsndfile swig bzip2 bottleneck gcc_linux-64=8.4.0 gxx_linux-64=8.4.0 --yes

# Install the python lib
pip install -r requirements.txt

# Install the auto_log
pushd tools/extras
bash install_autolog.sh
popd

# Install the ctcdecoder
pushd paddlespeech/s2t/decoders/ctcdecoder/swig
bash -e setup.sh
popd

# Install the python_speech_features
pushd third_party
bash -e install.sh
popd
