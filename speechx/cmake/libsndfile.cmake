include(FetchContent)

# https://github.com/pongasoft/vst-sam-spl-64/blob/master/libsndfile.cmake
# https://github.com/popojan/goban/blob/master/CMakeLists.txt#L38
# https://github.com/ddiakopoulos/libnyquist/blob/master/CMakeLists.txt

if(LIBSNDFILE_ROOT_DIR)
  # instructs FetchContent to not download or update but use the location instead
  set(FETCHCONTENT_SOURCE_DIR_LIBSNDFILE ${LIBSNDFILE_ROOT_DIR})
else()
  set(FETCHCONTENT_SOURCE_DIR_LIBSNDFILE "")
endif()

set(LIBSNDFILE_GIT_REPO "https://github.com/libsndfile/libsndfile.git" CACHE STRING "libsndfile git repository url" FORCE)
set(LIBSNDFILE_GIT_TAG 1.0.31 CACHE STRING "libsndfile git tag" FORCE)

FetchContent_Declare(libsndfile
      GIT_REPOSITORY    ${LIBSNDFILE_GIT_REPO}
      GIT_TAG           ${LIBSNDFILE_GIT_TAG}
      GIT_CONFIG        advice.detachedHead=false
#      GIT_SHALLOW       true
      CONFIGURE_COMMAND ""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
      )

FetchContent_GetProperties(libsndfile)
if(NOT libsndfile_POPULATED)
  if(FETCHCONTENT_SOURCE_DIR_LIBSNDFILE)
    message(STATUS "Using libsndfile from local ${FETCHCONTENT_SOURCE_DIR_LIBSNDFILE}")
  else()
    message(STATUS "Fetching libsndfile ${LIBSNDFILE_GIT_REPO}/tree/${LIBSNDFILE_GIT_TAG}")
  endif()
  FetchContent_Populate(libsndfile)
endif()

set(LIBSNDFILE_ROOT_DIR ${libsndfile_SOURCE_DIR})
set(LIBSNDFILE_INCLUDE_DIR "${libsndfile_BINARY_DIR}/src")

function(libsndfile_build)
  option(BUILD_PROGRAMS "Build programs" OFF)
  option(BUILD_EXAMPLES "Build examples" OFF)
  option(BUILD_TESTING "Build examples" OFF)
  option(ENABLE_CPACK "Enable CPack support" OFF)
  option(ENABLE_PACKAGE_CONFIG "Generate and install package config file" OFF)
  option(BUILD_REGTEST "Build regtest" OFF)
  # finally we include libsndfile itself
  add_subdirectory(${libsndfile_SOURCE_DIR} ${libsndfile_BINARY_DIR} EXCLUDE_FROM_ALL)
  # copying .hh for c++ support
  #file(COPY "${libsndfile_SOURCE_DIR}/src/sndfile.hh" DESTINATION ${LIBSNDFILE_INCLUDE_DIR})
endfunction()

libsndfile_build()

include_directories(${LIBSNDFILE_INCLUDE_DIR})