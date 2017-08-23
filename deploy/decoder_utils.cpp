#include <limits>
#include <algorithm>
#include <cmath>
#include "decoder_utils.h"

size_t get_utf8_str_len(const std::string& str) {
    size_t str_len = 0;
    for (char c : str) {
        str_len += ((c & 0xc0) != 0x80);
    }
    return str_len;
}
