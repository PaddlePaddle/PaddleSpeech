// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

#ifdef INCLUDE_KALDI
#include "paddleaudio/src/pybind/kaldi/kaldi_feature.h"
#include "paddleaudio/third_party/kaldi-native-fbank/csrc/feature-fbank.h"
#endif

#ifdef INCLUDE_SOX
#include "paddleaudio/src/pybind/sox/io.h"
#include "paddleaudio/src/pybind/sox/effects.h"
#endif

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

// `tl::optional` 
#ifdef INCLUDE_SOX
namespace pybind11 { namespace detail {
   template <typename T>
   struct type_caster<tl::optional<T>> : optional_caster<tl::optional<T>> {};
}}
#endif

PYBIND11_MODULE(_paddleaudio, m) {
#ifdef INCLUDE_SOX
    m.def("get_info_file",
          &paddleaudio::sox_io::get_info_file,
          "Get metadata of audio file.");
    // support obj later
    m.def("get_info_fileobj",
          &paddleaudio::sox_io::get_info_fileobj,
          "Get metadata of audio in file object.");
    m.def("load_audio_fileobj",
          &paddleaudio::sox_io::load_audio_fileobj,
          "Load audio from file object.");
    m.def("save_audio_fileobj",
          &paddleaudio::sox_io::save_audio_fileobj,
          "Save audio to file obj.");
          
    // sox io
     m.def("sox_io_get_info", &paddleaudio::sox_io::get_info_file);
     m.def(
         "sox_io_load_audio_file",
         &paddleaudio::sox_io::load_audio_file);
     m.def(
         "sox_io_save_audio_file",
         &paddleaudio::sox_io::save_audio_file);
    
     // sox utils
     m.def("sox_utils_set_seed", &paddleaudio::sox_utils::set_seed);
     m.def(
         "sox_utils_set_verbosity",
         &paddleaudio::sox_utils::set_verbosity);
     m.def(
         "sox_utils_set_use_threads",
         &paddleaudio::sox_utils::set_use_threads);
     m.def(
         "sox_utils_set_buffer_size",
         &paddleaudio::sox_utils::set_buffer_size);
     m.def(
         "sox_utils_list_effects",
         &paddleaudio::sox_utils::list_effects);
     m.def(
         "sox_utils_list_read_formats",
         &paddleaudio::sox_utils::list_read_formats);
     m.def(
         "sox_utils_list_write_formats",
         &paddleaudio::sox_utils::list_write_formats);
     m.def(
         "sox_utils_get_buffer_size",
         &paddleaudio::sox_utils::get_buffer_size);

     // effect
     m.def("apply_effects_fileobj",
           &paddleaudio::sox_effects::apply_effects_fileobj,
           "Decode audio data from file-like obj and apply effects.");
     m.def("sox_effects_initialize_sox_effects",
       &paddleaudio::sox_effects::initialize_sox_effects);
     m.def(
         "sox_effects_shutdown_sox_effects",
         &paddleaudio::sox_effects::shutdown_sox_effects);
     m.def(
         "sox_effects_apply_effects_tensor",
         &paddleaudio::sox_effects::apply_effects_tensor);
     m.def(
         "sox_effects_apply_effects_file",
         &paddleaudio::sox_effects::apply_effects_file);
#endif

#ifdef INCLUDE_KALDI
    m.def("ComputeFbank", &paddleaudio::kaldi::ComputeFbank, "compute fbank");
    //py::class_<kaldi::PitchExtractionOptions>(m, "PitchExtractionOptions")
        //.def(py::init<>())
        //.def_readwrite("samp_freq", &kaldi::PitchExtractionOptions::samp_freq)
        //.def_readwrite("frame_shift_ms", &kaldi::PitchExtractionOptions::frame_shift_ms)
        //.def_readwrite("frame_length_ms", &kaldi::PitchExtractionOptions::frame_length_ms)
        //.def_readwrite("preemph_coeff", &kaldi::PitchExtractionOptions::preemph_coeff)
        //.def_readwrite("min_f0", &kaldi::PitchExtractionOptions::min_f0)
        //.def_readwrite("max_f0", &kaldi::PitchExtractionOptions::max_f0)
        //.def_readwrite("soft_min_f0", &kaldi::PitchExtractionOptions::soft_min_f0)
        //.def_readwrite("penalty_factor", &kaldi::PitchExtractionOptions::penalty_factor)
        //.def_readwrite("lowpass_cutoff", &kaldi::PitchExtractionOptions::lowpass_cutoff)
        //.def_readwrite("resample_freq", &kaldi::PitchExtractionOptions::resample_freq)
        //.def_readwrite("delta_pitch", &kaldi::PitchExtractionOptions::delta_pitch)
        //.def_readwrite("nccf_ballast", &kaldi::PitchExtractionOptions::nccf_ballast)
        //.def_readwrite("lowpass_filter_width", &kaldi::PitchExtractionOptions::lowpass_filter_width)
        //.def_readwrite("upsample_filter_width", &kaldi::PitchExtractionOptions::upsample_filter_width)
        //.def_readwrite("max_frames_latency", &kaldi::PitchExtractionOptions::max_frames_latency)
        //.def_readwrite("frames_per_chunk", &kaldi::PitchExtractionOptions::frames_per_chunk)
        //.def_readwrite("simulate_first_pass_online", &kaldi::PitchExtractionOptions::simulate_first_pass_online)
        //.def_readwrite("recompute_frame", &kaldi::PitchExtractionOptions::recompute_frame)
        //.def_readwrite("nccf_ballast_online", &kaldi::PitchExtractionOptions::nccf_ballast_online)
        //.def_readwrite("snip_edges", &kaldi::PitchExtractionOptions::snip_edges);
    //m.def("ComputeKaldiPitch", &paddleaudio::kaldi::ComputeKaldiPitch, "compute kaldi pitch");
    py::class_<knf::FrameExtractionOptions>(m, "FrameExtractionOptions")
        .def(py::init<>())            
        .def_readwrite("samp_freq", &knf::FrameExtractionOptions::samp_freq)
        .def_readwrite("frame_shift_ms", &knf::FrameExtractionOptions::frame_shift_ms)            
        .def_readwrite("frame_length_ms", &knf::FrameExtractionOptions::frame_length_ms)
        .def_readwrite("dither", &knf::FrameExtractionOptions::dither)            
        .def_readwrite("preemph_coeff", &knf::FrameExtractionOptions::preemph_coeff)            
        .def_readwrite("remove_dc_offset", &knf::FrameExtractionOptions::remove_dc_offset)            
        .def_readwrite("window_type", &knf::FrameExtractionOptions::window_type)
        .def_readwrite("round_to_power_of_two", &knf::FrameExtractionOptions::round_to_power_of_two)           
        .def_readwrite("blackman_coeff", &knf::FrameExtractionOptions::blackman_coeff)          
        .def_readwrite("snip_edges", &knf::FrameExtractionOptions::snip_edges)
        .def_readwrite("max_feature_vectors", &knf::FrameExtractionOptions::max_feature_vectors);
    py::class_<knf::MelBanksOptions>(m, "MelBanksOptions")
        .def(py::init<>())
        .def_readwrite("num_bins", &knf::MelBanksOptions::num_bins)
        .def_readwrite("low_freq", &knf::MelBanksOptions::low_freq)
        .def_readwrite("high_freq", &knf::MelBanksOptions::high_freq)
        .def_readwrite("vtln_low", &knf::MelBanksOptions::vtln_low)
        .def_readwrite("vtln_high", &knf::MelBanksOptions::vtln_high)
        .def_readwrite("debug_mel", &knf::MelBanksOptions::debug_mel)
        .def_readwrite("htk_mode", &knf::MelBanksOptions::htk_mode);

    py::class_<paddleaudio::kaldi::FbankOptions>(m, "FbankOptions")
        .def(py::init<>())
        .def_readwrite("use_energy", &paddleaudio::kaldi::FbankOptions::use_energy)
        .def_readwrite("energy_floor", &paddleaudio::kaldi::FbankOptions::energy_floor)
        .def_readwrite("raw_energy", &paddleaudio::kaldi::FbankOptions::raw_energy)
        .def_readwrite("htk_compat", &paddleaudio::kaldi::FbankOptions::htk_compat)
        .def_readwrite("use_log_fbank", &paddleaudio::kaldi::FbankOptions::use_log_fbank)
        .def_readwrite("use_power", &paddleaudio::kaldi::FbankOptions::use_power);
#endif

}
