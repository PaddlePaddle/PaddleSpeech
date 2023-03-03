
#include "base/glog_utils.h"

namespace google {
    void InitGoogleLogging(const char* name) {
        LOG(INFO) << "dummpy InitGoogleLogging.";
    }

    void InstallFailureSignalHandler(){
        LOG(INFO) << "dummpy InstallFailureSignalHandler.";
    }
} // namespace google
