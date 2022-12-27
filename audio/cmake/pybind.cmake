#the pybind11 is from:https://github.com/pybind/pybind11
# Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

SET(PYBIND_ZIP "v2.10.0.zip")
SET(LOCAL_PYBIND_ZIP ${FETCHCONTENT_BASE_DIR}/${PYBIND_ZIP})
SET(PYBIND_SRC ${FETCHCONTENT_BASE_DIR}/pybind11)
SET(DOWNLOAD_URL "https://paddleaudio.bj.bcebos.com/build/v2.10.0.zip")
SET(PYBIND_TIMEOUT 600 CACHE STRING "Timeout in seconds when downloading pybind.")

IF(NOT EXISTS ${LOCAL_PYBIND_ZIP})
    FILE(DOWNLOAD ${DOWNLOAD_URL}
      ${LOCAL_PYBIND_ZIP}
      TIMEOUT ${PYBIND_TIMEOUT}
      STATUS ERR
      SHOW_PROGRESS
    )

    IF(ERR EQUAL 0)
        MESSAGE(STATUS "download pybind success")
    ELSE()
        MESSAGE(FATAL_ERROR "download pybind fail")
    ENDIF()
ENDIF()

IF(NOT EXISTS ${PYBIND_SRC})
    EXECUTE_PROCESS(
      COMMAND ${CMAKE_COMMAND} -E tar xfz ${LOCAL_PYBIND_ZIP}
       WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}
       RESULT_VARIABLE tar_result
    )

    file(RENAME ${FETCHCONTENT_BASE_DIR}/pybind11-2.10.0 ${PYBIND_SRC})

  IF (tar_result MATCHES 0)
      MESSAGE(STATUS "unzip pybind success")
  ELSE()
      MESSAGE(FATAL_ERROR "unzip pybind fail")
  ENDIF()

ENDIF()

include_directories(${PYBIND_SRC}/include)
