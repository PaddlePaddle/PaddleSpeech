include(FetchContent)
include(ExternalProject)

set(OpenBLAS_SOURCE_DIR ${fc_patch}/OpenBLAS-src)
set(OpenBLAS_PREFIX ${fc_patch}/OpenBLAS-prefix)

# ######################################################################################################################
# OPENBLAS  https://github.com/lattice/quda/blob/develop/CMakeLists.txt#L575
# ######################################################################################################################
enable_language(Fortran)

include(FortranCInterface)

# # Clang doesn't have a Fortran compiler in its suite (yet),
# # so detect libraries for gfortran; we need equivalents to
# # libgfortran and libquadmath, which are implicitly
# # linked by flags in CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES
# include(FindGFortranLibs REQUIRED)
# # Add directory containing libgfortran and libquadmath to
# # linker. Should also contain libgomp, if not using
# # Intel OpenMP runtime
# link_directories(${GFORTRAN_LIBRARIES_DIR})
# # gfortan dir in the docker.
# link_directories(/usr/local/gcc-8.2/lib64)
# # if you are working with C and Fortran
# FortranCInterface_VERIFY()

# # if you are working with C++ and Fortran
# FortranCInterface_VERIFY(CXX)


#TODO: switch to CPM
include(GNUInstallDirs)
ExternalProject_Add(
    OPENBLAS
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
    GIT_TAG v0.3.10
    GIT_SHALLOW YES
    PREFIX ${OpenBLAS_PREFIX}
    SOURCE_DIR  ${OpenBLAS_SOURCE_DIR}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> 
    CMAKE_GENERATOR "Unix Makefiles")


# https://cmake.org/cmake/help/latest/module/ExternalProject.html?highlight=externalproject_get_property#external-project-definition
ExternalProject_Get_Property(OPENBLAS INSTALL_DIR)
set(OpenBLAS_INSTALL_PREFIX ${INSTALL_DIR})
add_library(openblas STATIC IMPORTED)
add_dependencies(openblas OPENBLAS)
set_target_properties(openblas PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES Fortran)
# ${CMAKE_INSTALL_LIBDIR}  lib
set_target_properties(openblas PROPERTIES IMPORTED_LOCATION ${OpenBLAS_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libopenblas.a)


# https://cmake.org/cmake/help/latest/command/install.html?highlight=cmake_install_libdir#installing-targets
# ${CMAKE_INSTALL_LIBDIR}  lib
# ${CMAKE_INSTALL_INCLUDEDIR}  include
link_directories(${OpenBLAS_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
include_directories(${OpenBLAS_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/openblas)
