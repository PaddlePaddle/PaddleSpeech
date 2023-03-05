cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

if(ANDROID)
else() #Unix
    add_subdirectory(glog)
endif()