#pragma once

#include "decoder/ctc_beam_search_opt.h"
#include "decoder/ctc_tlg_decoder.h"
#include "frontend/feature_pipeline.h"

DECLARE_int32(nnet_decoder_chunk);
DECLARE_int32(num_left_chunks);
DECLARE_double(ctc_weight);
DECLARE_double(rescoring_weight);
DECLARE_double(reverse_weight);
DECLARE_int32(nbest);
DECLARE_int32(blank);
DECLARE_double(acoustic_scale);
DECLARE_string(word_symbol_table);

namespace ppspeech {

struct DecodeOptions {
    // chunk_size is the frame number of one chunk after subsampling.
    // e.g. if subsample rate is 4 and chunk_size = 16, the frames in
    // one chunk are 67=16*4 + 3, stride is 64=16*4
    int chunk_size{16};
    int num_left_chunks{-1};

    // final_score = rescoring_weight * rescoring_score + ctc_weight *
    // ctc_score;
    // rescoring_score = left_to_right_score * (1 - reverse_weight) +
    // right_to_left_score * reverse_weight
    // Please note the concept of ctc_scores
    // in the following two search methods are different. For
    // CtcPrefixBeamSerch,
    // it's a sum(prefix) score + context score For CtcWfstBeamSerch, it's a
    // max(viterbi) path score + context score So we should carefully set
    // ctc_weight accroding to the search methods.
    float ctc_weight{0.0};
    float rescoring_weight{1.0};
    float reverse_weight{0.0};

    // CtcEndpointConfig ctc_endpoint_opts;
    CTCBeamSearchOptions ctc_prefix_search_opts{};
    TLGDecoderOptions tlg_decoder_opts{};

    static DecodeOptions InitFromFlags() {
        DecodeOptions decoder_opts;
        decoder_opts.chunk_size = FLAGS_nnet_decoder_chunk;
        decoder_opts.num_left_chunks = FLAGS_num_left_chunks;
        decoder_opts.ctc_weight = FLAGS_ctc_weight;
        decoder_opts.rescoring_weight = FLAGS_rescoring_weight;
        decoder_opts.reverse_weight = FLAGS_reverse_weight;
        decoder_opts.ctc_prefix_search_opts.blank = FLAGS_blank;
        decoder_opts.ctc_prefix_search_opts.first_beam_size = FLAGS_nbest;
        decoder_opts.ctc_prefix_search_opts.second_beam_size = FLAGS_nbest;
        decoder_opts.ctc_prefix_search_opts.word_symbol_table = 
            FLAGS_word_symbol_table;
        decoder_opts.tlg_decoder_opts =
            ppspeech::TLGDecoderOptions::InitFromFlags();

        LOG(INFO) << "chunk_size: " << decoder_opts.chunk_size;
        LOG(INFO) << "num_left_chunks: " << decoder_opts.num_left_chunks;
        LOG(INFO) << "ctc_weight: " << decoder_opts.ctc_weight;
        LOG(INFO) << "rescoring_weight: " << decoder_opts.rescoring_weight;
        LOG(INFO) << "reverse_weight: " << decoder_opts.reverse_weight;
        LOG(INFO) << "blank: " << FLAGS_blank;
        LOG(INFO) << "first_beam_size: " << FLAGS_nbest;
        LOG(INFO) << "second_beam_size: " << FLAGS_nbest;
        return decoder_opts;
    }
};

struct RecognizerResource {
    // decodable opt 
    kaldi::BaseFloat acoustic_scale{1.0};

    FeaturePipelineOptions feature_pipeline_opts{};
    ModelOptions model_opts{};
    DecodeOptions decoder_opts{};
    std::shared_ptr<NnetBase> nnet;

    static RecognizerResource InitFromFlags() {
        RecognizerResource resource;
        resource.acoustic_scale = FLAGS_acoustic_scale;
        LOG(INFO) << "acoustic_scale: " << resource.acoustic_scale;

        resource.feature_pipeline_opts =
            ppspeech::FeaturePipelineOptions::InitFromFlags();
        resource.feature_pipeline_opts.assembler_opts.fill_zero = false;
        LOG(INFO) << "u2 need fill zero be false: "
                  << resource.feature_pipeline_opts.assembler_opts.fill_zero;
        resource.model_opts = ppspeech::ModelOptions::InitFromFlags();
        resource.decoder_opts = ppspeech::DecodeOptions::InitFromFlags();
        #ifndef USE_ONNX
            resource.nnet.reset(new U2Nnet(resource.model_opts));
        #else
            if (resource.model_opts.with_onnx_model){
                resource.nnet.reset(new U2OnnxNnet(resource.model_opts));
            } else {
                resource.nnet.reset(new U2Nnet(resource.model_opts));
            }
        #endif
        return resource;
    }
};

} //namespace ppspeech