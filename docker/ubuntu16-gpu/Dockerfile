FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu16.04

RUN echo "deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial universe \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates universe \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial multiverse \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates multiverse \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security universe \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security multiverse" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y inetutils-ping wget vim curl cmake git sox libsndfile1 libpng12-dev \
    libpng-dev swig libzip-dev openssl bc libflac* libgdk-pixbuf2.0-dev libpango1.0-dev libcairo2-dev \
    libgtk2.0-dev pkg-config zip unzip zlib1g-dev libreadline-dev libbz2-dev liblapack-dev libjpeg-turbo8-dev \
    sudo lrzsz libsqlite3-dev libx11-dev libsm6 apt-utils libopencv-dev libavcodec-dev libavformat-dev \
    libswscale-dev locales liblzma-dev python-lzma m4 libxext-dev strace libibverbs-dev libpcre3 libpcre3-dev \
    build-essential libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev xz-utils \
    libfreetype6-dev libxslt1-dev libxml2-dev libgeos-3.5.0 libgeos-dev && apt-get install -y --allow-downgrades \
    --allow-change-held-packages libnccl2 libnccl-dev && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
    && /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && \
    cd /usr/lib/x86_64-linux-gnu && ln -s libcudnn.so.8 libcudnn.so && \
    cd /usr/local/cuda-11.2/targets/x86_64-linux/lib  && ln -s libcublas.so.11.4.1.1043 libcublas.so && \
    ln -s libcusolver.so.11.1.0.152 libcusolver.so && ln -s libcusparse.so.11 libcusparse.so

RUN echo "set meta-flag on" >> /etc/inputrc && echo "set convert-meta off" >> /etc/inputrc && \
    locale-gen en_US.UTF-8 && /sbin/ldconfig -v && groupadd -g 10001 paddle && \
    useradd -m -s /bin/bash -N -u 10001 paddle -g paddle && chmod g+w /etc/passwd && \
    echo "paddle ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 LANGUAGE=en_US.UTF-8 TZ=Asia/Shanghai

RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz && tar xvf Python-3.6.15.tgz && \
    cd Python-3.6.15 && ./configure --prefix=/home/paddle/python3.6 && make -j8 && make install && \
    rm -rf ../Python-3.6.15 ../Python-3.6.15.tgz && chown -R paddle:paddle /home/paddle/python3.6

RUN wget https://mirrors.sjtug.sjtu.edu.cn/gnu/gmp/gmp-6.1.0.tar.bz2 && tar xvf gmp-6.1.0.tar.bz2 && \
    cd gmp-6.1.0 && ./configure --prefix=/usr/local && make -j8 && make install && \
    rm -rf ../gmp-6.1.0.tar.bz2 ../gmp-6.1.0 && cd .. && \
    wget https://www.mpfr.org/mpfr-3.1.4/mpfr-3.1.4.tar.bz2 && tar xvf mpfr-3.1.4.tar.bz2 && cd mpfr-3.1.4 && \
    ./configure --prefix=/usr/local && make -j8 && make install && rm -rf ../mpfr-3.1.4.tar.bz2 ../mpfr-3.1.4 && \
    cd .. && wget https://mirrors.sjtug.sjtu.edu.cn/gnu/mpc/mpc-1.0.3.tar.gz && tar xvf mpc-1.0.3.tar.gz && \
    cd mpc-1.0.3 && ./configure --prefix=/usr/local && make -j8 && make install && \
    rm -rf ../mpc-1.0.3.tar.gz ../mpc-1.0.3 && cd .. && \
    wget http://www.mirrorservice.org/sites/sourceware.org/pub/gcc/infrastructure/isl-0.18.tar.bz2 && \
    tar xvf isl-0.18.tar.bz2 && cd isl-0.18 && ./configure --prefix=/usr/local && make -j8 && make install \
    && rm -rf ../isl-0.18.tar.bz2 ../isl-0.18 && cd .. && \
    wget http://mirrors.ustc.edu.cn/gnu/gcc/gcc-8.2.0/gcc-8.2.0.tar.gz --no-check-certificate && \
    tar xvf gcc-8.2.0.tar.gz && cd gcc-8.2.0 && UNSET LIBRARY_PATH && ./configure --prefix=/home/paddle/gcc82 \
    --enable-threads=posix --disable-checking --disable-multilib --enable-languages=c,c++ --with-gmp=/usr/local \
    --with-mpfr=/usr/local --with-mpc=/usr/local --with-isl=/usr/local && make -j8 && make install && \
    rm -rf ../gcc-8.2.0.tar.gz ../gcc-8.2.0 && chown -R paddle:paddle /home/paddle/gcc82

WORKDIR /home/paddle
USER paddle
ENV PATH=/home/paddle/python3.6/bin:/home/paddle/gcc82/bin \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

RUN mkdir ~/.pip && echo "" > ~/.pip/pip.conf && echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url=https://mirror.baidu.com/pypi/simple" >> ~/.pip/pip.conf && \
    echo "trusted-host=mirror.baidu.com" >> ~/.pip/pip.conf && \
    python3 -m pip install --upgrade pip && \
    pip3 install paddlepaddle-gpu==2.3.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html 

RUN git clone https://github.com/PaddlePaddle/PaddleSpeech.git && cd PaddleSpeech && \
    pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -e .[develop] -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    cp -f /home/paddle/gcc82/lib64/libstdc++.so.6.0.25 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 
