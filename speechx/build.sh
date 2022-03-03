#!/usr/bin/env bash

boost_SOURCE_DIR=$PWD/fc_patch/boost-src
if [ ! -d ${boost_SOURCE_DIR} ]; then
  wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz 
  tar xzfv boost_1_75_0.tar.gz
  mkdir -p $PWD/fc_patch
  mv boost_1_75_0 ${boost_SOURCE_DIR} 
  cd ${boost_SOURCE_DIR}
  bash ./bootstrap.sh
  ./b2
  cd -
  echo -e "\n"
fi

mkdir build
cd build

cmake .. -DBOOST_ROOT:STRING=${boost_SOURCE_DIR}

make

cd -
