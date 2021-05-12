#ifndef _CHUNK_H_
#define _CHUNK_H_

#include <cmath>

#include "word.h"

namespace rmmseg {
/**
 * A chunk stores 3 (or less) successive words.
 */
struct Chunk {
    int total_length() const {
        int len = 0;
        for (int i = 0; i < n; ++i) len += std::abs(words[i]->length);
        // if (words[i]->length == -1) /* tmp word */
        //    len += 1;
        // else
        //    len += words[i]->length;
        return len;
    }
    double average_length() const { return ((double)total_length()) / n; }
    double variance() const {
        double avg = average_length();
        double sqr_sum = 0;
        double tmp;
        for (int i = 0; i < n; ++i) {
            tmp = std::abs(words[i]->length);
            // if (tmp == -1)
            //    tmp = 1;
            tmp = tmp - avg;
            sqr_sum += tmp * tmp;
        }
        return std::sqrt(sqr_sum);
    }
    int degree_of_morphemic_freedom() const {
        int sum = 0;
        for (int i = 0; i < n; ++i) sum += words[i]->freq;
        return sum;
    }


    int n;
    Word *words[3];
};
}

#endif /* _CHUNK_H_ */
