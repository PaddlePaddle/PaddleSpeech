FROM registry.baidubce.com/paddlepaddle/paddle:2.2.2
LABEL maintainer="paddlesl@baidu.com"

RUN git clone https://github.com/PaddlePaddle/PaddleSpeech.git /home/PaddleSpeech 
RUN pip3 uninstall mccabe -y; exit 0;
RUN pip3 install multiprocess==0.70.12
RUN pip3 install importlib-metadata==4.2.0

RUN cd /home/PaddleSpeech/audio
RUN python setup.py bdist_wheel
RUN pip install dist/*.whl

RUN cd /home/PaddleSpeech
RUN python setup.py bdist_wheel
RUN pip install dist/*.whl

WORKDIR /home/PaddleSpeech/
