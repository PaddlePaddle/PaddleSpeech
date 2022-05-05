// Copyright (c) 2022 PaddlePaddle Wenet Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "base/common.h"

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"

#include "decoder/recognizer.h"
#include "frontend/audio/feature_pipeline.h"

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

namespace ppspeech {
class ConnectionHandler {
  public:
    ConnectionHandler(tcp::socket&& socket,
                      const RecognizerResource& recognizer_resource_);
    void operator()();

  private:
    void OnSpeechStart();
    void OnSpeechEnd();
    void OnText(const std::string& message);
    void OnFinish();
    void OnSpeechData(const beast::flat_buffer& buffer);
    void OnError(const std::string& message);
    void OnPartialResult(const std::string& result);
    void OnFinalResult(const std::string& result);
    void DecodeThreadFunc();
    std::string SerializeResult(bool finish);

    bool continuous_decoding_ = false;
    int nbest_ = 1;
    websocket::stream<tcp::socket> ws_;
    RecognizerResource recognizer_resource_;

    bool got_start_tag_ = false;
    bool got_end_tag_ = false;
    // When endpoint is detected, stop recognition, and stop receiving data.
    bool stop_recognition_ = false;
    std::shared_ptr<ppspeech::Recognizer> recognizer_ = nullptr;
    std::shared_ptr<std::thread> decode_thread_ = nullptr;
};

class WebSocketServer {
  public:
    WebSocketServer(int port, const RecognizerResource& recognizer_resource)
        : port_(port), recognizer_resource_(recognizer_resource) {}

    void Start();

  private:
    int port_;
    RecognizerResource recognizer_resource_;
    // The io_context is required for all I/O
    asio::io_context ioc_{1};
};

}  // namespace ppspeech
