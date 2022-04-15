// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "base/common.h"

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"

namespace ppspeech {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

class WebSocketClient {
  public:
    WebSocketClient(const std::string& host, int port);

    void SendTextData(const std::string& data);
    void SendBinaryData(const void* data, size_t size);
    void ReadLoopFunc();
    void Close();
    void Join();
    void SendStartSignal();
    void SendEndSignal();
    bool done() const { return done_; }

  private:
    void Connect();
    std::string host_;
    int port_;
    bool done_ = false;
    asio::io_context ioc_;
    websocket::stream<tcp::socket> ws_{ioc_};
    std::unique_ptr<std::thread> t_{nullptr};
};