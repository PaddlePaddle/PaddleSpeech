# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

set(CBLAS_PREFIX_DIR ${THIRD_PARTY_PATH}/openblas)
set(CBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
set(CBLAS_REPOSITORY https://github.com/xianyi/OpenBLAS.git)
set(CBLAS_TAG v0.3.10)

if(NOT WIN32)
  set(CBLAS_LIBRARIES
      "${CBLAS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
      CACHE FILEPATH "openblas library." FORCE)
  set(CBLAS_INC_DIR
      "${CBLAS_INSTALL_DIR}/include"
      CACHE PATH "openblas include directory." FORCE)
  set(OPENBLAS_CC
      "${CMAKE_C_COMPILER} -Wno-unused-but-set-variable -Wno-unused-variable")

  if(APPLE)
    set(OPENBLAS_CC "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
  endif()
  set(OPTIONAL_ARGS "")
  set(COMMON_ARGS "")

  if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
      set(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
    endif()
    set(COMMON_ARGS CC=${OPENBLAS_CC} NO_SHARED=1 libs)
  endif()

  ExternalProject_Add(
    OPENBLAS
    GIT_REPOSITORY ${CBLAS_REPOSITORY}
    GIT_TAG ${CBLAS_TAG}
    GIT_SHALLOW YES
    PREFIX ${CBLAS_PREFIX_DIR}
    INSTALL_DIR ${CBLAS_INSTALL_DIR}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make -j${NPROC} ${COMMON_ARGS} ${OPTIONAL_ARGS}
    INSTALL_COMMAND make install PREFIX=<INSTALL_DIR>
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_BYPRODUCTS ${CBLAS_LIBRARIES})

    ExternalProject_Get_Property(OPENBLAS INSTALL_DIR)
    set(OpenBLAS_INSTALL_PREFIX ${INSTALL_DIR})
    add_library(openblas STATIC IMPORTED)
    add_dependencies(openblas OPENBLAS)
    set_target_properties(openblas PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES Fortran)
    set_target_properties(openblas PROPERTIES IMPORTED_LOCATION ${OpenBLAS_INSTALL_PREFIX}/lib/libopenblas.a)

    link_directories(${OpenBLAS_INSTALL_PREFIX}/lib)
    include_directories(${OpenBLAS_INSTALL_PREFIX}/include)

    set(OPENBLAS_LIBRARIES
        ${OpenBLAS_INSTALL_PREFIX}/lib/libopenblas.a
    )

    add_library(libopenblas INTERFACE)
    add_dependencies(libopenblas openblas)
    target_include_directories(libopenblas INTERFACE ${OpenBLAS_INSTALL_PREFIX}/include/openblas)
    target_link_libraries(libopenblas INTERFACE ${OPENBLAS_LIBRARIES})
else()
  set(CBLAS_LIBRARIES
      "${CBLAS_INSTALL_DIR}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
      CACHE FILEPATH "openblas library." FORCE)
  set(CBLAS_INC_DIR
      "${CBLAS_INSTALL_DIR}/include/openblas"
      CACHE PATH "openblas include directory." FORCE)
  ExternalProject_Add(
    extern_openblas
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY ${CBLAS_REPOSITORY}
    GIT_TAG ${CBLAS_TAG}
    PREFIX ${CBLAS_PREFIX_DIR}
    INSTALL_DIR ${CBLAS_INSTALL_DIR}
    BUILD_IN_SOURCE 0
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_C_COMPILER=clang-cl
               -DCMAKE_CXX_COMPILER=clang-cl
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${CBLAS_INSTALL_DIR}
               -DCMAKE_BUILD_TYPE=Release #${THIRD_PARTY_BUILD_TYPE}
               -DCMAKE_MT=mt
               -DUSE_THREAD=OFF
               -DBUILD_WITHOUT_LAPACK=NO
               -DCMAKE_Fortran_COMPILER=flang
               -DNOFORTRAN=0
               -DDYNAMIC_ARCH=ON
               #${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${CBLAS_INSTALL_DIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=Release #${THIRD_PARTY_BUILD_TYPE}
    # ninja need to know where openblas.lib comes from
    BUILD_BYPRODUCTS ${CBLAS_LIBRARIES})
  set(OPENBLAS_SHARED_LIB
      ${CBLAS_INSTALL_DIR}/bin/openblas${CMAKE_SHARED_LIBRARY_SUFFIX})

  add_library(openblas INTERFACE)
  add_dependencies(openblas extern_openblas)
  include_directories(${CBLAS_INC_DIR})
  link_libraries(${CBLAS_LIBRARIES})
endif()

