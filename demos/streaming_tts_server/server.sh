#!/bin/bash

# http server
paddlespeech_server start --config_file ./conf/tts_online_application.yaml &> tts.http.log &


# websocket server
paddlespeech_server start --config_file ./conf/tts_online_ws_application.yaml &> tts.ws.log &


