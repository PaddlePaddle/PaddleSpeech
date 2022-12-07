
add_subdirectory(third_party)
add_subdirectory(src)

if (APPLE) 
  file(COPY ${GFORTRAN_LIBRARIES_DIR}/libgcc_s.1.1.dylib
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
endif(APPLE)

if (UNIX AND NOT APPLE)
  file(COPY ${GFORTRAN_LIBRARIES_DIR}/libgfortran.so.5
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib FOLLOW_SYMLINK_CHAIN)

  file(COPY ${GFORTRAN_LIBRARIES_DIR}/libquadmath.so.0
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib FOLLOW_SYMLINK_CHAIN)

  file(COPY ${GFORTRAN_LIBRARIES_DIR}/libgcc_s.so.1
          DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib FOLLOW_SYMLINK_CHAIN)
endif()
