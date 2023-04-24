// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//               2022 PaddlePaddle Authors
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


#include "websocket/websocket_server.h"

#include "base/common.h"
#include "boost/json/src.hpp"

namespace json = boost::json;

namespace ppspeech {

ConnectionHandler::ConnectionHandler(
    tcp::socket&& socket, const RecognizerResource& recognizer_resource)
    : ws_(std::move(socket)), recognizer_resource_(recognizer_resource) {}

void ConnectionHandler::OnSpeechStart() {
    recognizer_ = std::make_shared<Recognizer>(recognizer_resource_);
    // Start decoder thread
    decode_thread_ = std::make_shared<std::thread>(
        &ConnectionHandler::DecodeThreadFunc, this);
    got_start_tag_ = true;
    LOG(INFO) << "Server: Received speech start signal, start reading speech";
    json::value rv = {{"status", "ok"}, {"type", "server_ready"}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
}

void ConnectionHandler::OnSpeechEnd() {
    LOG(INFO) << "Server: Received speech end signal";
    if (recognizer_ != nullptr) {
        recognizer_->SetFinished();
    }
    got_end_tag_ = true;
}

void ConnectionHandler::OnFinalResult(const std::string& result) {
    LOG(INFO) << "Server: Final result: " << result;
    json::value rv = {
        {"status", "ok"}, {"type", "final_result"}, {"result", result}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
}

void ConnectionHandler::OnFinish() {
    // Send finish tag
    json::value rv = {{"status", "ok"}, {"type", "speech_end"}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
}

void ConnectionHandler::OnSpeechData(const beast::flat_buffer& buffer) {
    // Read binary PCM data
    int num_samples = buffer.size() / sizeof(int16_t);
    kaldi::Vector<kaldi::BaseFloat> pcm_data(num_samples);
    const int16_t* pdata = static_cast<const int16_t*>(buffer.data().data());
    for (int i = 0; i < num_samples; i++) {
        pcm_data(i) = static_cast<float>(*pdata);
        pdata++;
    }
    VLOG(2) << "Server: Received " << num_samples << " samples";
    LOG(INFO) << "Server: Received " << num_samples << " samples";
    CHECK(recognizer_ != nullptr);
    recognizer_->Accept(pcm_data);

    std::string partial_result = recognizer_->GetPartialResult();

    json::value rv = {{"status", "ok"},
                      {"type", "partial_result"},
                      {"result", partial_result}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
}

void ConnectionHandler::DecodeThreadFunc() {
    try {
        while (true) {
            recognizer_->Decode();
            if (recognizer_->IsFinished()) {
                LOG(INFO) << "Server: enter finish";
                recognizer_->Decode();
                LOG(INFO) << "Server: finish";
                std::string result = recognizer_->GetFinalResult();
                OnFinalResult(result);
                OnFinish();
                stop_recognition_ = true;
                break;
            }
        }
    } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
    }
}

void ConnectionHandler::OnError(const std::string& message) {
    json::value rv = {{"status", "failed"}, {"message", message}};
    ws_.text(true);
    ws_.write(asio::buffer(json::serialize(rv)));
    // Close websocket
    ws_.close(websocket::close_code::normal);
}

void ConnectionHandler::OnText(const std::string& message) {
    json::value v = json::parse(message);
    if (v.is_object()) {
        json::object obj = v.get_object();
        if (obj.find("signal") != obj.end()) {
            json::string signal = obj["signal"].as_string();
            if (signal == "start") {
                OnSpeechStart();
            } else if (signal == "end") {
                OnSpeechEnd();
            } else {
                OnError("Unexpected signal type");
            }
        } else {
            OnError("Wrong message header");
        }
    } else {
        OnError("Wrong protocol");
    }
}

void ConnectionHandler::operator()() {
    try {
        // Accept the websocket handshake
        ws_.accept();
        for (;;) {
            // This buffer will hold the incoming message
            beast::flat_buffer buffer;
            // Read a message
            ws_.read(buffer);
            if (ws_.got_text()) {
                std::string message = beast::buffers_to_string(buffer.data());
                LOG(INFO) << "Server: Text: " << message;
                OnText(message);
                if (got_end_tag_) {
                    break;
                }
            } else {
                if (!got_start_tag_) {
                    OnError("Start signal is expected before binary data");
                } else {
                    if (stop_recognition_) {
                        break;
                    }
                    OnSpeechData(buffer);
                }
            }
        }

        LOG(INFO) << "Server: finished to wait for decoding thread join.";
        if (decode_thread_ != nullptr) {
            decode_thread_->join();
        }
    } catch (beast::system_error const& se) {
        // This indicates that the session was closed
        if (se.code() != websocket::error::closed) {
            if (decode_thread_ != nullptr) {
                decode_thread_->join();
            }
            OnSpeechEnd();
            LOG(ERROR) << se.code().message();
        }
    } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
    }
}

void WebSocketServer::Start() {
    try {
        auto const address = asio::ip::make_address("0.0.0.0");
        tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
        for (;;) {
            // This will receive the new connection
            tcp::socket socket{ioc_};
            // Block until we get a connection
            acceptor.accept(socket);
            // Launch the session, transferring ownership of the socket
            ConnectionHandler handler(std::move(socket), recognizer_resource_);
            std::thread t(std::move(handler));
            t.detach();
        }
    } catch (const std::exception& e) {
        LOG(FATAL) << e.what();
    }
}

}  // namespace ppspeech
