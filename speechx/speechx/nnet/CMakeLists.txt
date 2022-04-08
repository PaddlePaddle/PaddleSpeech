project(nnet)

add_library(nnet STATIC
  decodable.cc
  paddle_nnet.cc
)
target_link_libraries(nnet absl::strings)