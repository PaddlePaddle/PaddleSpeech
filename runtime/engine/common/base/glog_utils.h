#pragma once

#include "base/common.h"

namespace google {
void InitGoogleLogging(const char* name);

void InstallFailureSignalHandler();
}  // namespace google