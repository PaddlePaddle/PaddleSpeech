sudo nvidia-docker run --privileged  --net=host --ipc=host -it --rm -v $PWD:/paddle --name=paddle_demo_docker registry.baidubce.com/paddlepaddle/paddle:2.2.2 /bin/bash
