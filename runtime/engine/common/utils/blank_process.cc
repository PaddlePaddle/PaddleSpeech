#include "utils/blank_process.h"

namespace ppspeech {

std::string RemoveBlk(const std::string& str) {
    std::string out = "";
    int ptr_in = 0;    //  the pointer of input string (for traversal)
    int end = str.size();
    int ptr_out = -1;  //  the pointer of output string (last char)
    while (ptr_in != end) {
        while (ptr_in != end && str[ptr_in] == ' ') {
            ptr_in += 1;
        }
        if (ptr_in == end) 
            return out;
        if (ptr_out != -1 && isalpha(str[ptr_in]) && isalpha(str[ptr_out]) && str[ptr_in-1] == ' ')
            // add a space when the last and current chars are in English and there have space(s) between them
            out += ' ';
        out += str[ptr_in];
        ptr_out = ptr_in;
        ptr_in += 1;
    }
    return out;
}

}  // namespace ppspeech