#pragma once

#include "base/common.h"

#include "frontend/audio/feature_pipeline.h"
#include "decoder/ctc_beam_search_decoder.h"
#include "decoder/ctc_tlg_decoder.h"

DEFINE_string(cmvn_file, "", "read cmvn");
DEFINE_double(streaming_chunk, 0.36, "streaming feature chunk size");
DEFINE_string(model_path, "avg_1.jit.pdmodel", "paddle nnet model");
DEFINE_string(params_path, "avg_1.jit.pdiparams", "paddle nnet model param");
DEFINE_string(word_symbol_table, "words.txt", "word symbol table");
DEFINE_string(graph_path, "TLG", "decoder graph");
DEFINE_double(acoustic_scale, 1.0, "acoustic scale");
DEFINE_int32(max_active, 7500, "max active");
DEFINE_double(beam, 15.0, "decoder beam");
DEFINE_double(lattice_beam, 7.5, "decoder beam");
DEFINE_int32(receptive_field_length,
             7,
             "receptive field of two CNN(kernel=5) downsampling module.");
DEFINE_int32(downsampling_rate,
             4,
             "two CNN(kernel=5) module downsampling rate.");
DEFINE_string(model_output_names,
              "save_infer_model/scale_0.tmp_1,save_infer_model/"
              "scale_1.tmp_1,save_infer_model/scale_2.tmp_1,save_infer_model/"
              "scale_3.tmp_1",
              "model output names");
DEFINE_string(model_cache_names, "5-1-1024,5-1-1024", "model cache names");

namespace ppspeech {
    // todo refactor later
    FeaturePipelineOptions InitFeaturePipelineOptions() {
       FeaturePipelineOptions opts; 
       opts.cmvn_file = FLAGS_cmvn_file;
       opts.linear_spectrogram_opts.streaming_chunk = FLAGS_streaming_chunk;
       kaldi::FrameExtractionOptions frame_opts;
       frame_opts.frame_length_ms = 20;
       frame_opts.frame_shift_ms = 10;
       frame_opts.remove_dc_offset = false;
       frame_opts.window_type = "hanning";
       frame_opts.preemph_coeff = 0.0;
       frame_opts.dither = 0.0;
       opts.linear_spectrogram_opts.frame_opts = frame_opts;
       opts.feature_cache_opts.frame_chunk_size = FLAGS_receptive_field_length;
       opts.feature_cache_opts.frame_chunk_stride = FLAGS_downsampling_rate;
       return opts;
    }

    ModelOptions InitModelOptions() {
       ModelOptions model_opts;
       model_opts.model_path = FLAGS_model_path;
       model_opts.params_path = FLAGS_params_path;
       model_opts.cache_shape = FLAGS_model_cache_names;
       model_opts.output_names = FLAGS_model_output_names;
       return model_opts;
    }

    TLGDecoderOptions InitDecoderOptions() {
       TLGDecoderOptions decoder_opts;
       decoder_opts.word_symbol_table = FLAGS_word_symbol_table;
       decoder_opts.fst_path = FLAGS_graph_path;
       decoder_opts.opts.max_active = FLAGS_max_active;
       decoder_opts.opts.beam = FLAGS_beam;
       decoder_opts.opts.lattice_beam = FLAGS_lattice_beam;
       return decoder_opts;
    }
    
    RecognizerResource InitRecognizerResoure() {
       RecognizerResource resource;
       resource.acoustic_scale = FLAGS_acoustic_scale;
       resource.feature_pipeline_opts = InitFeaturePipelineOptions();
       resource.model_opts = InitModelOptions();
       resource.tlg_opts = InitDecoderOptions();
       return resource;
    }
    

}