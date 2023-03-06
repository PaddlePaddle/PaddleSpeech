#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "paddle_api.h"

using namespace paddle::lite_api;

class Predictor {
public:
    bool Init(const std::string &AMModelPath, const std::string &VOCModelPath, int cpuThreadNum, const std::string &cpuPowerMode) {
        // Release model if exists
        ReleaseModel();

        AM_predictor_ = LoadModel(AMModelPath, cpuThreadNum, cpuPowerMode);
        if (AM_predictor_ == nullptr) {
            return false;
        }
        VOC_predictor_ = LoadModel(VOCModelPath, cpuThreadNum, cpuPowerMode);
        if (VOC_predictor_ == nullptr) {
            return false;
        }

        return true;
    }

    ~Predictor() {
        ReleaseModel();
        ReleaseWav();
    }

    std::shared_ptr<PaddlePredictor> LoadModel(const std::string &modelPath, int cpuThreadNum, const std::string &cpuPowerMode) {
        if (modelPath.empty()) {
            return nullptr;
        }

        // 设置MobileConfig
        MobileConfig config;
        config.set_model_from_file(modelPath);
        config.set_threads(cpuThreadNum);

        if (cpuPowerMode == "LITE_POWER_HIGH") {
            config.set_power_mode(PowerMode::LITE_POWER_HIGH);
        } else if (cpuPowerMode == "LITE_POWER_LOW") {
            config.set_power_mode(PowerMode::LITE_POWER_LOW);
        } else if (cpuPowerMode == "LITE_POWER_FULL") {
            config.set_power_mode(PowerMode::LITE_POWER_FULL);
        } else if (cpuPowerMode == "LITE_POWER_NO_BIND") {
            config.set_power_mode(PowerMode::LITE_POWER_NO_BIND);
        } else if (cpuPowerMode == "LITE_POWER_RAND_HIGH") {
            config.set_power_mode(PowerMode::LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode == "LITE_POWER_RAND_LOW") {
            config.set_power_mode(PowerMode::LITE_POWER_RAND_LOW);
        } else {
            std::cerr << "Unknown cpu power mode!" << std::endl;
            return nullptr;
        }

        return CreatePaddlePredictor<MobileConfig>(config);
    }

    void ReleaseModel() {
        AM_predictor_ = nullptr;
        VOC_predictor_ = nullptr;
    }

    bool RunModel(const std::vector<float> &phones) {
        if (!IsLoaded()) {
            return false;
        }

        // 计时开始
        auto start = std::chrono::system_clock::now();

        // 执行推理
        VOCOutputToWav(GetAMOutput(phones));

        // 计时结束
        auto end = std::chrono::system_clock::now();

        // 计算用时
        std::chrono::duration<float> duration = end - start;
        inference_time_ = duration.count() * 1000; // 单位：毫秒

        return true;
    }

    std::unique_ptr<const Tensor> GetAMOutput(const std::vector<float> &phones) {
        auto phones_handle = AM_predictor_->GetInput(0);
        phones_handle->Resize({static_cast<int64_t>(phones.size())});
        phones_handle->CopyFromCpu(phones.data());
        AM_predictor_->Run();

        // 获取输出Tensor
        auto am_output_handle = AM_predictor_->GetOutput(0);
        // 打印输出Tensor的shape
        std::cout << "AM Output shape: ";
        auto shape = am_output_handle->shape();
        for (auto s : shape) {
            std::cout << s << ", ";
        }
        std::cout << std::endl;

        // 获取输出Tensor的数据
        auto am_output_data = am_output_handle->mutable_data<float>();
        return am_output_handle;
    }

    void VOCOutputToWav(std::unique_ptr<const Tensor> &&input) {
        auto mel_handle = VOC_predictor_->GetInput(0);
        // [?, 80]
        auto dims = input->shape();
        mel_handle->Resize(dims);
        auto am_output_data = input->mutable_data<float>();
        mel_handle->CopyFromCpu(am_output_data);
        VOC_predictor_->Run();

        // 获取输出Tensor
        auto voc_output_handle = VOC_predictor_->GetOutput(0);
        // 打印输出Tensor的shape
        std::cout << "VOC Output shape: ";
        auto shape = voc_output_handle->shape();
        for (auto s : shape) {
            std::cout << s << ", ";
        }
        std::cout << std::endl;

        // 获取输出Tensor的数据
        int64_t output_size = 1;
        for (auto dim : voc_output_handle->shape()) {
            output_size *= dim;
        }
        wav_.resize(output_size);
        auto output_data = voc_output_handle->mutable_data<float>();
        std::copy_n(output_data, output_size, wav_.data());
    }

    bool IsLoaded() {
        return AM_predictor_ != nullptr && VOC_predictor_ != nullptr;
    }

    float GetInferenceTime() {
        return inference_time_;
    }

    const std::vector<float> & GetWav() {
        return wav_;
    }

    void ReleaseWav() {
        wav_.clear();
    }

    struct WavHeader {
        // RIFF 头
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t size = 0;
        char wave[4] = {'W', 'A', 'V', 'E'};

        // FMT 头
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_size = 16;
        uint16_t audio_format = 3;
        uint16_t num_channels = 1;

        // 如果播放速度和音调异常，请修改采样率
        // 常见采样率：16000, 24000, 32000, 44100, 48000, 96000
        uint32_t sample_rate = 24000;

        uint32_t byte_rate = 64000;
        uint16_t block_align = 4;
        uint16_t bits_per_sample = 32;

        // DATA 头
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size = 0;
    };

    bool WriteWavToFile(const std::string &wavPath) {
        std::ofstream fout(wavPath, std::ios::binary);
        if (!fout.is_open()) {
            return false;
        }

        // 写入头信息
        WavHeader header;
        header.size = sizeof(header) - 8;
        header.data_size = wav_.size() * sizeof(float);
        header.byte_rate = header.sample_rate * header.num_channels * header.bits_per_sample / 8;
        header.block_align = header.num_channels * header.bits_per_sample / 8;
        fout.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // 写入wav数据
        fout.write(reinterpret_cast<const char*>(wav_.data()), header.data_size);

        fout.close();
        return true;
    }

private:
    float inference_time_ = 0;
    std::shared_ptr<PaddlePredictor> AM_predictor_ = nullptr;
    std::shared_ptr<PaddlePredictor> VOC_predictor_ = nullptr;
    std::vector<float> wav_;
};
