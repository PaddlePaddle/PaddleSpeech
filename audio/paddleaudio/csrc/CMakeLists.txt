find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

function(define_extension name sources libraries)
  add_library(${name} SHARED ${sources})
  target_include_directories(
    ${name} PRIVATE ${PROJECT_SOURCE_DIR} ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
  target_link_libraries(${name} ${libraries})
  set_target_properties(${name} PROPERTIES PREFIX "")
  install(
    TARGETS ${name}
    LIBRARY DESTINATION .
    )
endfunction()

if(BUILD_SOX)
  set(
    EXTENSION_SOURCES
    pybind/pybind.cpp
    pybind/sox/io.cpp
    pybind/sox/utils.cpp
    )

  set(
    LINK_LIBRARIES
    libsox
    )

  define_extension(
    _paddleaudio
    "${EXTENSION_SOURCES}"
    "${LINK_LIBRARIES}"
    )
endif()
