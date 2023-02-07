include(FetchContent)
set(Boost_DEBUG ON)

set(Boost_PREFIX_DIR ${fc_patch}/boost)
set(Boost_SOURCE_DIR ${fc_patch}/boost-src)

FetchContent_Declare(
  Boost
  URL      https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
  PREFIX            ${Boost_PREFIX_DIR}
  SOURCE_DIR        ${Boost_SOURCE_DIR}
)

execute_process(COMMAND bootstrap.sh WORKING_DIRECTORY ${Boost_SOURCE_DIR})
execute_process(COMMAND b2 WORKING_DIRECTORY ${Boost_SOURCE_DIR})

FetchContent_MakeAvailable(Boost)

message(STATUS "boost src dir: ${Boost_SOURCE_DIR}")
message(STATUS "boost inc dir: ${Boost_INCLUDE_DIR}")
message(STATUS "boost bin dir: ${Boost_BINARY_DIR}")

set(BOOST_ROOT ${Boost_SOURCE_DIR})
message(STATUS "boost root dir: ${BOOST_ROOT}")

include_directories(${Boost_SOURCE_DIR})