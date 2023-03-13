#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "paddle_api.h"

using namespace paddle::lite_api;

class PredictorInterface {
public:
    virtual ~PredictorInterface() = 0;
    virtual bool Init(
            const std::string &AcousticModelPath,
            const std::string &VocoderPath,
            PowerMode cpuPowerMode,
            int cpuThreadNum,
            // WAV采样率（必须与模型输出匹配）
            // 如果播放速度和音调异常，请修改采样率
            // 常见采样率：16000, 24000, 32000, 44100, 48000, 96000
            uint32_t wavSampleRate
    ) = 0;
    virtual std::shared_ptr<PaddlePredictor> LoadModel(const std::string &modelPath, int cpuThreadNum, PowerMode cpuPowerMode) = 0;
    virtual void ReleaseModel() = 0;
    virtual bool RunModel(const std::vector<int64_t> &phones) = 0;
    virtual std::unique_ptr<const Tensor> GetAcousticModelOutput(const std::vector<int64_t> &phones) = 0;
    virtual std::unique_ptr<const Tensor> GetVocoderOutput(std::unique_ptr<const Tensor> &&amOutput) = 0;
    virtual void VocoderOutputToWav(std::unique_ptr<const Tensor> &&vocOutput) = 0;
    virtual void SaveFloatWav(float *floatWav, int64_t size) = 0;
    virtual bool IsLoaded() = 0;
    virtual float GetInferenceTime() = 0;
    virtual int GetWavSize() = 0;
    // 获取WAV持续时间（单位：毫秒）
    virtual float GetWavDuration() = 0;
    // 获取RTF（合成时间 / 音频时长）
    virtual float GetRTF() = 0;
    virtual void ReleaseWav() = 0;
    virtual bool WriteWavToFile(const std::string &wavPath) = 0;
};

PredictorInterface::~PredictorInterface() {}

// WavDataType: WAV数据类型
// 可在 int16_t 和 float 之间切换，
// 用于生成 16-bit PCM 或 32-bit IEEE float 格式的 WAV
template<typename WavDataType>
class Predictor : public PredictorInterface {
public:
    virtual bool Init(
            const std::string &AcousticModelPath,
            const std::string &VocoderPath,
            PowerMode cpuPowerMode,
            int cpuThreadNum,
            // WAV采样率（必须与模型输出匹配）
            // 如果播放速度和音调异常，请修改采样率
            // 常见采样率：16000, 24000, 32000, 44100, 48000, 96000
            uint32_t wavSampleRate
    ) override {
        // Release model if exists
        ReleaseModel();

        acoustic_model_predictor_ = LoadModel(AcousticModelPath, cpuThreadNum, cpuPowerMode);
        if (acoustic_model_predictor_ == nullptr) {
            return false;
        }
        vocoder_predictor_ = LoadModel(VocoderPath, cpuThreadNum, cpuPowerMode);
        if (vocoder_predictor_ == nullptr) {
            return false;
        }

        wav_sample_rate_ = wavSampleRate;

        return true;
    }

    virtual ~Predictor() {
        ReleaseModel();
        ReleaseWav();
    }

    virtual std::shared_ptr<PaddlePredictor> LoadModel(const std::string &modelPath, int cpuThreadNum, PowerMode cpuPowerMode) override {
        if (modelPath.empty()) {
            return nullptr;
        }

        // 设置MobileConfig
        MobileConfig config;
        config.set_model_from_file(modelPath);
        config.set_threads(cpuThreadNum);
        config.set_power_mode(cpuPowerMode);

        return CreatePaddlePredictor<MobileConfig>(config);
    }

    virtual void ReleaseModel() override {
        acoustic_model_predictor_ = nullptr;
        vocoder_predictor_ = nullptr;
    }

    virtual bool RunModel(const std::vector<int64_t> &phones) override {
        if (!IsLoaded()) {
            return false;
        }

        // 计时开始
        auto start = std::chrono::system_clock::now();

        // 执行推理
        VocoderOutputToWav(GetVocoderOutput(GetAcousticModelOutput(phones)));

        // 计时结束
        auto end = std::chrono::system_clock::now();

        // 计算用时
        std::chrono::duration<float> duration = end - start;
        inference_time_ = duration.count() * 1000; // 单位：毫秒

        return true;
    }

    virtual std::unique_ptr<const Tensor> GetAcousticModelOutput(const std::vector<int64_t> &phones) override {
        auto phones_handle = acoustic_model_predictor_->GetInput(0);
        phones_handle->Resize({static_cast<int64_t>(phones.size())});
        phones_handle->CopyFromCpu(phones.data());
        acoustic_model_predictor_->Run();

        // 获取输出Tensor
        auto am_output_handle = acoustic_model_predictor_->GetOutput(0);
        // 打印输出Tensor的shape
        std::cout << "Acoustic Model Output shape: ";
        auto shape = am_output_handle->shape();
        for (auto s : shape) {
            std::cout << s << ", ";
        }
        std::cout << std::endl;

        return am_output_handle;
    }

    virtual std::unique_ptr<const Tensor> GetVocoderOutput(std::unique_ptr<const Tensor> &&amOutput) override {
        auto mel_handle = vocoder_predictor_->GetInput(0);
        // [?, 80]
        auto dims = amOutput->shape();
        mel_handle->Resize(dims);
        auto am_output_data = amOutput->mutable_data<float>();
        mel_handle->CopyFromCpu(am_output_data);
        vocoder_predictor_->Run();

        // 获取输出Tensor
        auto voc_output_handle = vocoder_predictor_->GetOutput(0);
        // 打印输出Tensor的shape
        std::cout << "Vocoder Output shape: ";
        auto shape = voc_output_handle->shape();
        for (auto s : shape) {
            std::cout << s << ", ";
        }
        std::cout << std::endl;

        return voc_output_handle;
    }

    virtual void VocoderOutputToWav(std::unique_ptr<const Tensor> &&vocOutput) override {
        // 获取输出Tensor的数据
        int64_t output_size = 1;
        for (auto dim : vocOutput->shape()) {
            output_size *= dim;
        }
        auto output_data = vocOutput->mutable_data<float>();

        SaveFloatWav(output_data, output_size);
    }

    virtual void SaveFloatWav(float *floatWav, int64_t size) override;

    virtual bool IsLoaded() override {
        return acoustic_model_predictor_ != nullptr && vocoder_predictor_ != nullptr;
    }

    virtual float GetInferenceTime() override {
        return inference_time_;
    }

    const std::vector<WavDataType> & GetWav() {
        return wav_;
    }

    virtual int GetWavSize() override {
        return wav_.size() * sizeof(WavDataType);
    }

    // 获取WAV持续时间（单位：毫秒）
    virtual float GetWavDuration() override {
        return static_cast<float>(GetWavSize()) / sizeof(WavDataType) / static_cast<float>(wav_sample_rate_) * 1000;
    }

    // 获取RTF（合成时间 / 音频时长）
    virtual float GetRTF() override {
        return GetInferenceTime() / GetWavDuration();
    }

    virtual void ReleaseWav() override {
        wav_.clear();
    }

    virtual bool WriteWavToFile(const std::string &wavPath) override {
        std::ofstream fout(wavPath, std::ios::binary);
        if (!fout.is_open()) {
            return false;
        }

        // 写入头信息
        WavHeader header;
        header.audio_format = GetWavAudioFormat();
        header.data_size = GetWavSize();
        header.size = sizeof(header) - 8 + header.data_size;
        header.sample_rate = wav_sample_rate_;
        header.byte_rate = header.sample_rate * header.num_channels * header.bits_per_sample / 8;
        header.block_align = header.num_channels * header.bits_per_sample / 8;
        fout.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // 写入wav数据
        fout.write(reinterpret_cast<const char*>(wav_.data()), header.data_size);

        fout.close();
        return true;
    }

protected:
    struct WavHeader {
        // RIFF 头
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t size = 0;
        char wave[4] = {'W', 'A', 'V', 'E'};

        // FMT 头
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_size = 16;
        uint16_t audio_format = 0;
        uint16_t num_channels = 1;
        uint32_t sample_rate = 0;
        uint32_t byte_rate = 0;
        uint16_t block_align = 0;
        uint16_t bits_per_sample = sizeof(WavDataType) * 8;

        // DATA 头
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size = 0;
    };

    enum WavAudioFormat {
        WAV_FORMAT_16BIT_PCM   = 1, // 16-bit PCM 格式
        WAV_FORMAT_32BIT_FLOAT = 3  // 32-bit IEEE float 格式
    };

protected:
    // 返回值通过模板特化由 WavDataType 决定
    inline uint16_t GetWavAudioFormat();

    inline float Abs(float number) {
        return (number < 0) ? -number : number;
    }

protected:
    float inference_time_ = 0;
    uint32_t wav_sample_rate_ = 0;
    std::vector<WavDataType> wav_;
    std::shared_ptr<PaddlePredictor> acoustic_model_predictor_ = nullptr;
    std::shared_ptr<PaddlePredictor> vocoder_predictor_ = nullptr;
};

template<>
uint16_t Predictor<int16_t>::GetWavAudioFormat() {
    return Predictor::WAV_FORMAT_16BIT_PCM;
}

template<>
uint16_t Predictor<float>::GetWavAudioFormat() {
    return Predictor::WAV_FORMAT_32BIT_FLOAT;
}

// 保存 16-bit PCM 格式 WAV
template<>
void Predictor<int16_t>::SaveFloatWav(float *floatWav, int64_t size) {
    wav_.resize(size);
    float maxSample = 0.01;
    // 寻找最大采样值
    for (int64_t i=0; i<size; i++) {
        float sample = Abs(floatWav[i]);
        if (sample > maxSample) {
            maxSample = sample;
        }
    }
    // 把采样值缩放到 int_16 范围
    for (int64_t i=0; i<size; i++) {
        wav_[i] = floatWav[i] * 32767.0f / maxSample;
    }
}

// 保存 32-bit IEEE float 格式 WAV
template<>
void Predictor<float>::SaveFloatWav(float *floatWav, int64_t size) {
    wav_.resize(size);
    std::copy_n(floatWav, size, wav_.data());
}
