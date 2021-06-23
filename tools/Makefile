PYTHON:= python3.7
.PHONY: all clean

all: virtualenv kenlm.done sox.done soxbindings.done

virtualenv:
	test -d venv || virtualenv -p $(PYTHON) venv
	touch venv/bin/activate

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
	rm -rf kenlm

kenlm.done:
	# Ubuntu 16.04 透過 apt 會安裝 boost 1.58.0
	# it seems that boost (1.54.0) requires higher version. After I switched to g++-5 it compiles normally.
	apt install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
	apt-get install -y gcc-5 g++-5 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
	test -d kenlm || wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
	mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j4 && make install
	cd kenlm && python setup.py install
	touch kenlm.done

sox.done:
	apt install -y libvorbis-dev libmp3lame-dev libmad-ocaml-dev
	test -d sox-14.4.2 || wget https://nchc.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz
	tar -xvzf sox-14.4.2.tar.gz -C .
	cd sox-14.4.2 && ./configure --prefix=/usr/ && make -j4 && make install
	touch sox.done

soxbindings.done:
	test -d soxbindings || git clone https://github.com/pseeth/soxbindings.git
	source venv/bin/activate; cd soxbindings && python setup.py install
	touch soxbindings.done