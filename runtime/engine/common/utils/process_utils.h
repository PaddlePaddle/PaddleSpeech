
#pragma once

#include <unistd.h>
#include <string>

namespace ppspeech {

pid_t GetPid() {
    return getpid();
}

void GetPrpocessName(const pid_t pid, std::string& name){
    char tmp[256];
    sprintf(tmp, "/proc/%d/cmdline", pid);
    
    FILE* f = fopen(tmp, "r");
    if(f){
        size_t size;
        size = fread(name.data(), sizeof(char), sizeof(tmp), f);
        if(size > 0){
            if ('\n' == tmp[size-1]){
                tmp
            }
        }
    }
}

} // namespace ppspeech