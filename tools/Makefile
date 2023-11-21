SHELL:= /bin/bash
PYTHON:= python3.8

CXX ?= g++
CC ?= gcc        # used for sph2pipe
# CXX = clang++  # Uncomment these lines...
# CC = clang     # ...to build with Clang.

WGET ?= wget --no-check-certificate

.PHONY: all clean 

all: apt.done kenlm.done mfa.done sctk.done


virtualenv.done:
	test -d venv || virtualenv -p $(PYTHON) venv
	touch virtualenv.done

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
	rm -rf kenlm


apt.done:
	apt update -y
	apt install -y bc flac jq vim tig tree sox pkg-config libsndfile1 libflac-dev libogg-dev libvorbis-dev libboost-dev swig python3-dev 
	echo "check_certificate = off" >> ~/.wgetrc
	touch apt.done


kenlm.done:
	# Ubuntu 16.04 透過 apt 會安裝 boost 1.58.0
	# it seems that boost (1.54.0) requires higher version. After I switched to g++-5 it compiles normally.
	apt install -y --allow-unauthenticated build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
	apt-get install -y gcc-5 g++-5 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50  && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 50
	test -d kenlm || $(WGET) -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
	rm -rf kenlm/build && mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j4 && make install
	cd kenlm && python3 setup.py install
	touch kenlm.done


mfa.done:
	test -d montreal-forced-aligner || $(WGET) https://paddlespeech.bj.bcebos.com/Parakeet/montreal-forced-aligner_linux.tar.gz
	tar xvf montreal-forced-aligner_linux.tar.gz
	touch mfa.done

openblas.done:
	bash extras/install_openblas.sh
	touch openblas.done

kaldi.done: apt.done openblas.done
	bash extras/install_kaldi.sh
	touch kaldi.done

sctk.done:
	./extras/install_sclite.sh
	touch sctk.done

srilm.done:
	./extras/install_liblbfgs.sh
	extras/install_srilm.sh
	touch srilm.done

######################
dev: python conda_packages.done sctk.done

# Use pip for paddle installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
USE_CONDA := 1
else
USE_CONDA :=
endif

python: activate_python.sh 

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

bc.done: activate_python.sh
	. ./activate_python.sh && { command -v bc || conda install -y bc -c conda-forge; }
	touch bc.done
cmake.done: activate_python.sh
	. ./activate_python.sh && { command -v cmake || conda install -y cmake; }
	touch cmake.done
flac.done: activate_python.sh
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	touch flac.done
ffmpeg.done: activate_python.sh
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
	touch ffmpeg.done
sox.done: activate_python.sh
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	touch sox.done
sndfile.done: activate_python.sh
	. ./activate_python.sh && { python3 -c "from ctypes.util import find_library as F; assert F('sndfile') is not None" || conda install -y libsndfile=1.0.28 -c conda-forge; }
	touch sndfile.done
ifneq ($(strip $(USE_CONDA)),)
conda_packages.done: bc.done cmake.done flac.done ffmpeg.done sox.done sndfile.done
else
conda_packages.done:
endif
	touch conda_packages.done
