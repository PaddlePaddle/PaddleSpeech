#.rst:
# FindGFortranLibs
# --------
#  https://github.com/Argonne-National-Laboratory/PIPS/blob/master/cmake/Modules/FindGFortranLibs.cmake
#  https://enccs.github.io/cmake-workshop/cxx-fortran/
#
# Find gcc Fortran compiler & library paths
#
# The module defines the following variables:
#
# ::
#
#
#   GFORTRANLIBS_FOUND - true if system has gfortran
#   LIBGFORTRAN_LIBRARIES - path to libgfortran
#   LIBQUADMATH_LIBRARIES - path to libquadmath
#   GFORTRAN_LIBARIES_DIR - directory containing libgfortran, libquadmath
#   GFORTRAN_INCLUDE_DIR - directory containing gfortran/gcc headers
#   LIBGOMP_LIBRARIES - path to libgomp
#   LIBGOMP_INCLUDE_DIR - directory containing omp.h header
#   GFORTRAN_VERSION_STRING - version of gfortran found
#
set(CMAKE_REQUIRED_QUIET ${LIBIOMP_FIND_QUIETLY})

if(NOT CMAKE_REQUIRED_QUIET)
  message(STATUS "Looking for gfortran related libraries...")
endif()

enable_language(Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")

  # Basically, call "gfortran -v" to dump compiler info to the string
  # GFORTRAN_VERBOSE_STR, which will be used to get necessary paths
  message(STATUS "Extracting library and header information by calling 'gfortran -v'...")
  execute_process(COMMAND "${CMAKE_Fortran_COMPILER}" "-v" ERROR_VARIABLE
    GFORTRAN_VERBOSE_STR RESULT_VARIABLE FLAG)

  # For debugging
  message(STATUS "'gfortran -v' returned:")
  message(STATUS "${GFORTRAN_VERBOSE_STR}")

  # Detect gfortran version
  string(REGEX MATCH "gcc version [^\t\n ]+" GFORTRAN_VER_STR "${GFORTRAN_VERBOSE_STR}")
  string(REGEX REPLACE "gcc version ([^\t\n ]+)" "\\1" GFORTRAN_VERSION_STRING "${GFORTRAN_VER_STR}")
  message(STATUS "Detected gfortran version ${GFORTRAN_VERSION_STRING}")
  unset(GFORTRAN_VER_STR)

  set(MATCH_REGEX "[^\t\n ]+[\t\n ]+")
  set(REPLACE_REGEX "([^\t\n ]+)")

  # Find architecture for compiler
  string(REGEX MATCH "Target: [^\t\n ]+"
    GFORTRAN_ARCH_STR "${GFORTRAN_VERBOSE_STR}")
  message(STATUS "Architecture string: ${GFORTRAN_ARCH_STR}")
  string(REGEX REPLACE "Target: ([^\t\n ]+)" "\\1"
    GFORTRAN_ARCH "${GFORTRAN_ARCH_STR}")
  message(STATUS "Detected gfortran architecture: ${GFORTRAN_ARCH}")
  unset(GFORTRAN_ARCH_STR)

  # Find install prefix, if it exists; if not, use default
  string(REGEX MATCH  "--prefix=[^\t\n ]+[\t\n ]+"
    GFORTRAN_PREFIX_STR "${GFORTRAN_VERBOSE_STR}")
  if(NOT GFORTRAN_PREFIX_STR)
    message(STATUS "Detected default gfortran prefix")
    set(GFORTRAN_PREFIX_DIR "/usr/local") # default prefix for gcc install
  else()
    string(REGEX REPLACE "--prefix=([^\t\n ]+)" "\\1"
      GFORTRAN_PREFIX_DIR "${GFORTRAN_PREFIX_STR}")
  endif()
  message(STATUS "Detected gfortran prefix: ${GFORTRAN_PREFIX_DIR}")
  unset(GFORTRAN_PREFIX_STR)

  # Find install exec-prefix, if it exists; if not, use default
  string(REGEX MATCH "--exec-prefix=[^\t\n ]+[\t\n ]+" "\\1"
    GFORTRAN_EXEC_PREFIX_STR "${GFORTRAN_VERBOSE_STR}")
  if(NOT GFORTRAN_EXEC_PREFIX_STR)
    message(STATUS "Detected default gfortran exec-prefix")
    set(GFORTRAN_EXEC_PREFIX_DIR "${GFORTRAN_PREFIX_DIR}")
  else()
    string(REGEX REPLACE "--exec-prefix=([^\t\n ]+)" "\\1"
      GFORTRAN_EXEC_PREFIX_DIR "${GFORTRAN_EXEC_PREFIX_STR}")
  endif()
  message(STATUS "Detected gfortran exec-prefix: ${GFORTRAN_EXEC_PREFIX_DIR}")
  UNSET(GFORTRAN_EXEC_PREFIX_STR)

  # Find library directory and include directory, if library directory specified
  string(REGEX MATCH "--libdir=[^\t\n ]+"
    GFORTRAN_LIB_DIR_STR "${GFORTRAN_VERBOSE_STR}")
  if(NOT GFORTRAN_LIB_DIR_STR)
    message(STATUS "Found --libdir flag -- not found")
    message(STATUS "Using default gfortran library & include directory paths")
    set(GFORTRAN_LIBRARIES_DIR
      "${GFORTRAN_EXEC_PREFIX_DIR}/lib/gcc/${GFORTRAN_ARCH}/${GFORTRAN_VERSION_STRING}")
    string(CONCAT GFORTRAN_INCLUDE_DIR "${GFORTRAN_LIBRARIES_DIR}" "/include")
  else()
    message(STATUS "Found --libdir flag -- yes")
    string(REGEX REPLACE "--libdir=([^\t\n ]+)" "\\1"
      GFORTRAN_LIBRARIES_DIR "${GFORTRAN_LIB_DIR_STR}")
    string(CONCAT GFORTRAN_INCLUDE_DIR "${GFORTRAN_LIBRARIES_DIR}" "/gcc/" "${GFORTRAN_ARCH}" "/" "${GFORTRAN_VERSION_STRING}" "/include")
  endif()
  message(STATUS "gfortran libraries path: ${GFORTRAN_LIBRARIES_DIR}")
  message(STATUS "gfortran include path dir: ${GFORTRAN_INCLUDE_DIR}")
  unset(GFORTRAN_LIB_DIR_STR)

  # There are lots of other build options for gcc & gfortran. For now, the
  # options implemented above should cover a lot of common use cases.

  # Clean up be deleting the output string from "gfortran -v"
  unset(GFORTRAN_VERBOSE_STR)

  # Find paths for libgfortran, libquadmath, libgomp
  # libgomp needed for OpenMP support without Clang
  find_library(LIBGFORTRAN_LIBRARIES NAMES gfortran libgfortran
    HINTS ${GFORTRAN_LIBRARIES_DIR})
  find_library(LIBQUADMATH_LIBRARIES NAMES quadmath libquadmath
    HINTS ${GFORTRAN_LIBRARIES_DIR})
  find_library(LIBGOMP_LIBRARIES NAMES gomp libgomp
    HINTS ${GFORTRAN_LIBRARIES_DIR})

  # Find OpenMP headers
  find_path(LIBGOMP_INCLUDE_DIR NAMES omp.h HINTS ${GFORTRAN_INCLUDE_DIR})

else()
  message(STATUS "CMAKE_Fortran_COMPILER_ID does not match 'GNU'!")
endif()

include(FindPackageHandleStandardArgs)

# Required: libgfortran, libquadmath, path for gfortran libraries
# Optional: libgomp, path for OpenMP headers, path for gcc/gfortran headers
find_package_handle_standard_args(GFortranLibs
  REQUIRED_VARS LIBGFORTRAN_LIBRARIES LIBQUADMATH_LIBRARIES GFORTRAN_LIBRARIES_DIR
  VERSION_VAR GFORTRAN_VERSION_STRING)

if(GFORTRANLIBS_FOUND)
  message(STATUS "Looking for gfortran libraries -- found")
  message(STATUS "gfortran version: ${GFORTRAN_VERSION_STRING}")
else()
  message(STATUS "Looking for gfortran libraries -- not found")
endif()

mark_as_advanced(LIBGFORTRAN_LIBRARIES LIBQUADMATH_LIBRARIES
  LIBGOMP_LIBRARIES LIBGOMP_INCLUDE_DIR
  GFORTRAN_LIBRARIES_DIR GFORTRAN_INCLUDE_DIR)
# FindGFortranLIBS.cmake ends here