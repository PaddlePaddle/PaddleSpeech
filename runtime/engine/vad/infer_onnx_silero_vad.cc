
#include "vad.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: infer_onnx_silero_vad path/to/model path/to/audio "
                     "run_option, "
                     "e.g ./infer_onnx_silero_vad silero_vad.onnx sample.wav"
                  << std::endl;
        return -1;
    }

    std::string model_file = argv[1];
    std::string audio_file = argv[2];

    int sr = 16000;
    Vad vad(model_file);
    // custom config, but must be set before init
    vad.SetConfig(sr, 32, 0.45f, 200, 0, 0);
    vad.Init();

    std::vector<float> inputWav;  // [0, 1]
    wav::WavReader wav_reader = wav::WavReader(audio_file);
    assert(wav_reader.sample_rate() == sr);


    auto num_samples = wav_reader.num_samples();
    inputWav.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
        inputWav[i] = wav_reader.data()[i] / 32768;
    }

    int window_size_samples = vad.WindowSizeSamples();
    for (int64_t j = 0; j < num_samples; j += window_size_samples) {
        auto start = j;
        auto end = start + window_size_samples >= num_samples
                       ? num_samples
                       : start + window_size_samples;
        auto current_chunk_size = end - start;

        std::vector<float> r{&inputWav[0] + start, &inputWav[0] + end};
        assert(r.size() == current_chunk_size);

        if (!vad.ForwardChunk(r)) {
            std::cerr << "Failed to inference while using model:"
                      << vad.ModelName() << "." << std::endl;
            return false;
        }

        Vad::State s = vad.Postprocess();
        std::cout << s << " ";
    }
    std::cout << std::endl;

    std::vector<std::map<std::string, float>> result = vad.GetResult();
    for (auto& res : result) {
        std::cout << "speak start: " << res["start"]
                  << " s, end: " << res["end"] << " s | ";
    }
    std::cout << "\b\b " << std::endl;

    vad.Reset();

    return 0;
}
