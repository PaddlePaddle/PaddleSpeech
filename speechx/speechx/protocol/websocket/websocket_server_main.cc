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
    resource.feature_pipeline_opts = ppspeech::InitFeaturePipelineOptions();

    ppspeech::ModelOptions model_opts;
    model_opts.model_path = FLAGS_model_path;
    model_opts.param_path = FLAGS_param_path;
    model_opts.cache_names = FLAGS_model_cache_names;
    model_opts.cache_shape = FLAGS_model_cache_shapes;
    model_opts.input_names = FLAGS_model_input_names;
    model_opts.output_names = FLAGS_model_output_names;
    model_opts.subsample_rate = FLAGS_downsampling_rate;
    resource.model_opts = model_opts;

    ppspeech::TLGDecoderOptions decoder_opts;
    decoder_opts.word_symbol_table = FLAGS_word_symbol_table;
    decoder_opts.fst_path = FLAGS_graph_path;
    decoder_opts.opts.max_active = FLAGS_max_active;
    decoder_opts.opts.beam = FLAGS_beam;
    decoder_opts.opts.lattice_beam = FLAGS_lattice_beam;

    resource.tlg_opts = decoder_opts;

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
