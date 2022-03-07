#include "frontend/raw_audio.h"
#include "kaldi/base/timer.h"

namespace ppspeech {

RawAudioSource::RawAudioSource(int buffer_size = 65536) 
    : finished_(false),
      data_length_(0),
      start_(0),
      timeout_(5) {
  ring_buffer_.resize(buffer_size);
} 

// todo length > buffer size, condition_var
bool RawAudioSource::AcceptWaveform(const VectorBase<BaseFloat>& data) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t idx = 0; idx < data.Dim(); ++idx) {
      ring_buffer_[idx % ring_buffer_.size()] = data(idx);
  }
  data_length_ += length;
}

// todo length > buffer size
//bool RawAudioSource::AcceptWaveform(BaseFloat* data, int length) {
  //std::lock_guard<std::mutex> lock(mutex_);
  //for (size_t idx = 0; idx < length; ++idx) {
      //ring_buffer_[idx % ring_buffer_.size()] = data[idx];
  //}
  //data_length_ += length;
  //finish_condition_.notify_one();
//}

bool RawAudioSource::Read(Vector<BaseFloat>* feats) {
  size_t chunk_size = feats->Dim();
  Timer timer;
  if (chunk_size > data_length_) {
    while (true) {
      int32 elapsed = static_cat<int32>(timer.Elapsed() * 1000);
      if (finished_ || > timeout_) {
        chunk_size = data_length_;
        feats->Resize(chunk_size);
        break;
      }
      sleep(1); 
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t idx = 0; idx < chunk_size; ++idx) {
    feats->Data()[idx] = ring_buffer_[idx];
  }
  data_length_ -= chunk_size;
  start_ = (start_ + chunk_size) % ring_buffer_.size();
  finish_condition_.notify_one();
}

//size_t RawAudioSource::GetDataLength() {
//  return data_length_;
//}

} // namespace ppspeech