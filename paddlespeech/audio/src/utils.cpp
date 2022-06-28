namespace paddleaudio {

namespace {

bool is_sox_available() {
#ifdef INCLUDE_SOX
  return true;
#else
  return false;
#endif
}

bool is_kaldi_available() {
#ifdef INCLUDE_KALDI
  return true;
#else
  return false;
#endif
}

// It tells whether paddleaudio was compiled with ffmpeg
// not the runtime availability.
bool is_ffmpeg_available() {
#ifdef USE_FFMPEG
  return true;
#else
  return false;
#endif
}

} // namespace

} // namespace paddleaudio