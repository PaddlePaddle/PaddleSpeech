project(websocket)

add_library(websocket STATIC
  websocket_server.cc
  websocket_client.cc
)
target_link_libraries(websocket PUBLIC frontend decoder nnet)
