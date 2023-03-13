include(FetchContent)

FetchContent_Declare(
  gflags
  URL      https://paddleaudio.bj.bcebos.com/build/gflag-2.2.2.zip 
  URL_HASH SHA256=19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5
)
FetchContent_MakeAvailable(gflags)

# openfst need
include_directories(${gflags_BINARY_DIR}/include)

install(FILES ${gflags_BINARY_DIR}/libgflags_nothreads.a DESTINATION lib)