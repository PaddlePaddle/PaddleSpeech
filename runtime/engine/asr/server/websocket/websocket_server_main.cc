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

#include "websocket/websocket_server.h"
#include "decoder/param.h"

DEFINE_int32(port, 8082, "websocket listening port");

ppspeech::RecognizerResource InitRecognizerResoure() {
    ppspeech::RecognizerResource resource;
    resource.acoustic_scale = FLAGS_acoustic_scale;
    resource.feature_pipeline_opts =
        ppspeech::FeaturePipelineOptions::InitFromFlags();
    resource.model_opts = ppspeech::ModelOptions::InitFromFlags();
    resource.tlg_opts = ppspeech::TLGDecoderOptions::InitFromFlags();
    return resource;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    ppspeech::RecognizerResource resource = InitRecognizerResoure();

    ppspeech::WebSocketServer server(FLAGS_port, resource);
    LOG(INFO) << "Listening at port " << FLAGS_port;
    server.Start();
    return 0;
}
