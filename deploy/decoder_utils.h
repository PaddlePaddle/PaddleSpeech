#ifndef DECODER_UTILS_H_
#define DECODER_UTILS_H_

#include <utility>

template <typename T1, typename T2>
bool pair_comp_first_rev(const std::pair<T1, T2> &a, const std::pair<T1, T2> &b)
{
    return a.first > b.first;
}

template <typename T1, typename T2>
bool pair_comp_second_rev(const std::pair<T1, T2> &a, const std::pair<T1, T2> &b)
{
    return a.second > b.second;
}

template <typename T>
T log_sum_exp(const T &x, const T &y)
{
    static T num_min = -std::numeric_limits<T>::max();
    if (x <= num_min) return y;
    if (y <= num_min) return x;
    T xmax = std::max(x, y);
    return std::log(std::exp(x-xmax) + std::exp(y-xmax)) + xmax;
}

// Get length of utf8 encoding string
// See: http://stackoverflow.com/a/4063229
size_t get_utf8_str_len(const std::string& str);

#endif // DECODER_UTILS_H
