FROM registry.baidubce.com/paddlepaddle/paddle:2.2.2
LABEL maintainer="paddlesl@baidu.com"

RUN git clone --depth 1 https://github.com/PaddlePaddle/PaddleSpeech.git /home/PaddleSpeech  
RUN pip3 uninstall mccabe -y ; exit 0;
RUN pip3 install multiprocess==0.70.12 importlib-metadata==4.2.0 dill==0.3.4

RUN cd /home/PaddleSpeech/audio
RUN python setup.py bdist_wheel

RUN cd /home/PaddleSpeech
RUN python setup.py bdist_wheel
RUN pip install audio/dist/*.whl dist/*.whl

WORKDIR /home/PaddleSpeech/
