include(FetchContent)


set(LIBSNDFILE_GIT_REPO "https://github.com/erikd/libsndfile" CACHE STRING "libsndfile git repository url" FORCE)
set(LIBSNDFILE_GIT_TAG 1.0.31 CACHE STRING "libsndfile git tag" FORCE)

FetchContent_Declare(libsndfile
      GIT_REPOSITORY    ${LIBSNDFILE_GIT_REPO}
      GIT_TAG           ${LIBSNDFILE_GIT_TAG}
      CMAKE_ARGS        "-G"Unix Makefiles""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
      )


set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(libsndfile)


set(LIBSNDFILE_ROOT_DIR ${libsndfile_SOURCE_DIR})
set(LIBSNDFILE_INCLUDE_DIR "${libsndfile_BINARY_DIR}/src")

#file(COPY "${libsndfile_SOURCE_DIR}/src/sndfile.hh" DESTINATION ${LIBSNDFILE_INCLUDE_DIR})

include_directories(${LIBSNDFILE_INCLUDE_DIR})