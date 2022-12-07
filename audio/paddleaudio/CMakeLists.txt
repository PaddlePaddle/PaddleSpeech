
add_subdirectory(third_party)
add_subdirectory(src)

if (APPLE) 
  file(COPY ${GFORTRAN_LIBRARIES_DIR}/libgcc_s.1.1.dylib
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
endif(APPLE)
