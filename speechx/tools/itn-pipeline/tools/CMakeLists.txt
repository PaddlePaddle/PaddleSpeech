cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet_text_processing VERSION 0.1)

include_directories(
${CMAKE_SOURCE_DIR}
)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

# Keep the same with openfst, -fPIC or -fpic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")

# third_party openfst
set(openfst_SOURCE_DIR ${fc_base}/openfst-src)
set(openfst_BINARY_DIR ${fc_base}/openfst-build)
set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix)
ExternalProject_Add(openfst
  URL               https://github.com/mjansche/openfst/archive/1.6.7.zip
  URL_HASH          SHA256=6ad450057ca4958de6c09358c66e2a137c5a1cbd2cb365b47866bf905fca72e2
  SOURCE_DIR        ${openfst_SOURCE_DIR}
  BINARY_DIR        ${openfst_BINARY_DIR}
  CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure --prefix=${openfst_PREFIX_DIR}
                     "--enable-compact-fsts=yes --enable-compress=yes --enable-const-fsts=yes --enable-far=yes "
                     "--enable-grm=yes --enable-linear-fsts=yes --enable-lookahead-fsts=yes --enable-mpdt=yes "
                     "--enable-ngram-fsts=yes --enable-pdt=yes "
                     "LIBS=-lpthread"
  BUILD_COMMAND     make -j 4
)
link_directories(${openfst_PREFIX_DIR}/lib)
include_directories(${openfst_SOURCE_DIR}/src/include)

# third_party thrax
set(thrax_SOURCE_DIR ${fc_base}/thrax-src)
set(thrax_BINARY_DIR ${fc_base}/thrax-build)
set(thrax_PREFIX_DIR ${fc_base}/thrax-subbuild/thrax-populate-prefix)
ExternalProject_Add(thrax
  URL               http://www.openfst.org/twiki/pub/GRM/ThraxDownload/thrax-1.2.9.tar.gz
  URL_HASH          SHA256=3cfbc003609d208d50d98659166086d4a6dd341ce6697e16370f5f7ae414f2b0
  SOURCE_DIR        ${thrax_SOURCE_DIR}
  BINARY_DIR        ${thrax_BINARY_DIR}
  CONFIGURE_COMMAND ${thrax_SOURCE_DIR}/configure --prefix=${thrax_PREFIX_DIR}
                     "LDFLAGS=-L${openfst_PREFIX_DIR}/lib"
                     "CXXFLAGS=-I${openfst_SOURCE_DIR}/src/include"
                     "LIBS=-lpthread"
  BUILD_COMMAND     make -j 4
)
add_dependencies(thrax openfst)
link_directories(${thrax_PREFIX_DIR}/lib)
include_directories(${thrax_SOURCE_DIR}/src/include)
