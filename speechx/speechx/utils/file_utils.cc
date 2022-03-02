#include "utils/file_utils.h"

namespace ppspeech {

bool ReadFileToVector(const std::string& filename,
                      std::vector<std::string>* vocabulary) {
    std::ifstream file_in(filename);
    if (!file_in) {
        std::cerr << "please input a valid file" << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file_in, line)) {
        vocabulary->emplace_back(line);
    }

    return true;
}

}