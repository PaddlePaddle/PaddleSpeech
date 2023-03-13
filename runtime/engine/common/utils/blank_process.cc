#include "utils/blank_process.h"

namespace ppspeech {

std::string BlankProcess(const std::string& str) {
    std::string out = "";
    int p = 0;
    int end = str.size();
    int q = -1;  //  last char of the output string
    while (p != end) {
        while (p != end && str[p] == ' ') {
            p += 1;
        }
        if (p == end) 
            return out;
        if (q != -1 && isalpha(str[p]) && isalpha(str[q]) && str[p-1] == ' ')
            // add a space when the last and current chars are in English and there have space(s) between them
            out += ' ';
        out += str[p];
        q = p;
        p += 1;
    }
    return out;
}

}  // namespace ppspeech