#!/usr/bin/env bash

# the build script had verified in the paddlepaddle docker image.
# please follow the instruction below to install PaddlePaddle image.
# https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html 
boost_SOURCE_DIR=$PWD/fc_patch/boost-src
if [ ! -d ${boost_SOURCE_DIR} ]; then wget -c https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz 
  tar xzfv boost_1_75_0.tar.gz
  mkdir -p $PWD/fc_patch
  mv boost_1_75_0 ${boost_SOURCE_DIR} 
  cd ${boost_SOURCE_DIR}
  bash ./bootstrap.sh
  ./b2
  cd -
  echo -e "\n"
fi

#rm -rf build
mkdir -p build
cd build

cmake .. -DBOOST_ROOT:STRING=${boost_SOURCE_DIR}
#cmake .. 

make -j

cd -
