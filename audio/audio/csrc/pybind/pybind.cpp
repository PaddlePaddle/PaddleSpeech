#include "pybind/sox/io.h"

PYBIND11_MODULE(_paddleaudio, m) {
    m.def("get_info_file",
          &paddleaudio::sox_io::get_info_file,
          "Get metadata of audio file.");
    m.def("get_info_fileobj",
          &paddleaudio::sox_io::get_info_fileobj,
          "Get metadata of audio in file object.");
}