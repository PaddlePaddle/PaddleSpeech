# project(websocket)

add_library(websocket STATIC
  websocket_server.cc
  websocket_client.cc
)
target_link_libraries(websocket PUBLIC frontend decoder nnet)

add_executable(websocket_server_main ${CMAKE_CURRENT_SOURCE_DIR}/websocket_server_main.cc)
target_include_directories(websocket_server_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(websocket_server_main PUBLIC fst websocket ${DEPS})

add_executable(websocket_client_main ${CMAKE_CURRENT_SOURCE_DIR}/websocket_client_main.cc)
target_include_directories(websocket_client_main PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
target_link_libraries(websocket_client_main PUBLIC fst websocket ${DEPS})
