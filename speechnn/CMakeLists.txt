cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speechnn VERSION 0.1)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/src CACHE PATH "Install path prefix." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# include file 
include(cmake/third_party.cmake)


set(CMAKE_VERBOSE_MAKEFILE on)
# set std-14
set(CMAKE_CXX_STANDARD 14)


# fc_patch dir
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_patch "fc_patch" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_patch})


###############################################################################
# Option Configurations
###############################################################################
# option configurations 
option(TEST_DEBUG "option for debug" OFF)


###############################################################################
# Add local library
###############################################################################
# system lib 
find_package()
# if dir have CmakeLists.txt 
add_subdirectory()
# if dir do not have CmakeLists.txt 
add_library(lib_name STATIC file.cc)
target_link_libraries(lib_name item0 item1)
add_dependencies(lib_name depend-target)


###############################################################################
# Library installation
###############################################################################
install()


###############################################################################
# Build binary file
###############################################################################
add_executable()
target_link_libraries()

