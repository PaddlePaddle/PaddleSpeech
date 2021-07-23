cmake_minimum_required(VERSION 3.14)
include(FetchContent)

FetchContent_Declare(
        libsndfile 
        GIT_REPOSITORY  https://github.com/libsndfile/libsndfile.git
        GIT_TAG         v1.0.30 # tag v1.0.30
)

FetchContent_GetProperties(libsndfile)
