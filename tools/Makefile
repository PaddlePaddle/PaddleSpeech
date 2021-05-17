PYTHON:= python3.8
.PHONY: all clean kenlm

all: virtualenv kenlm

virtualenv:
	test -d venv || virtualenv -p $(PYTHON) venv
	touch venv/bin/activate

kenlm:
	# Ubuntu 16.04 透過 apt 會安裝 boost 1.58.0
	# it seems that boost (1.54.0) requires higher version. After I switched to g++-5 it compiles normally.
	apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
	apt-get install -y gcc-5 g++-5 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
	test -d kenlm || wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
	mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j4 && make install
	cd kenlm && python setup.py install

clean:
	rm -fr venv
	find -iname "*.pyc" -delete

