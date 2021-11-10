#!/bin/bash 

apt install -y libvorbis-dev libmp3lame-dev libmad-ocaml-dev
test -d sox-14.4.2 || wget --no-check-certificate https://nchc.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz
tar -xvzf sox-14.4.2.tar.gz -C .
cd sox-14.4.2 && ./configure --prefix=/usr/ && make -j4 && make install