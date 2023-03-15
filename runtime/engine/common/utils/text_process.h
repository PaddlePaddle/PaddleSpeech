#include <string>
#include <vector>
#include <cctype>

namespace ppspeech {

std::string RemoveBlk(const std::string& str);

std::string AddBlk(const std::string& str);

std::string ReverseFrac(const std::string& str, 
                        const std::string& left_tag = "<tag>", 
                        const std::string& right_tag = "</tag>");

}  // namespace ppspeech