SHELL:= /bin/bash
PYTHON:= python3.7
.PHONY: all clean

all: virtualenv kenlm.done sox.done soxbindings.done mfa.done

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
	rm -rf kenlm/build && mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j4 && make install
	source venv/bin/activate; cd kenlm && python setup.py install
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

mfa.done:
	test -d montreal-forced-aligner || wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
	tar xvf montreal-forced-aligner_linux.tar.gz
	touch mfa.done


# Keep the existing target 'sclite' to avoid breaking the users who might have
# scripted it in.
.PHONY: sclite sctk_cleaned sctk_made

sclite sctk_made: sctk/.compiled

sctk/.compiled: sctk
	rm -f sctk/.compiled
	$(SCTK_MKENV) $(MAKE) -C sctk config
	$(SCTK_MKENV) $(MAKE) -C sctk all doc
	$(MAKE) -C sctk install
	touch sctk/.compiled

# The GitHub archive unpacks into SCTK-{40-character-long-hash}/
sctk: sctk-$(SCTK_GITHASH).tar.gz
	tar zxvf sctk-$(SCTK_GITHASH).tar.gz
	rm -rf sctk-$(SCTK_GITHASH) sctk
	mv SCTK-$(SCTK_GITHASH)* sctk-$(SCTK_GITHASH)
	ln -s sctk-$(SCTK_GITHASH) sctk
	touch sctk-$(SCTK_GITHASH).tar.gz

sctk-$(SCTK_GITHASH).tar.gz:
	if [ -d '$(DOWNLOAD_DIR)' ]; then \
	  cp -p '$(DOWNLOAD_DIR)/sctk-$(SCTK_GITHASH).tar.gz' .; \
	else \
	  $(WGET) -nv -T 10 -t 3 -O sctk-$(SCTK_GITHASH).tar.gz \
	    https://github.com/usnistgov/SCTK/archive/$(SCTK_GITHASH).tar.gz; \
	fi

sctk_cleaned:
	-for d in sctk/ sctk-*/; do \
	   [ ! -f $$d/.compiled ] || $(MAKE) -C $$d clean; \
	   rm -f $$d/.compiled; \
	done