// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// modified from https://github.com/Dounm/dlog
// modified form
// https://android.googlesource.com/platform/art/+/806defa/src/logging.h

#pragma once

#include <stdlib.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "base/common.h"
#include "base/macros.h"
#ifndef WITH_GLOG
#include "base/glog_utils.h"
#endif

DECLARE_int32(logtostderr);

namespace ppspeech {

namespace log {

enum Severity {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL,
    NUM_SEVERITIES,
};

class LogMessage {
  public:
    static void get_curr_proc_info(std::string* pid, std::string* proc_name);

    LogMessage(const char* file,
               int line,
               Severity level,
               bool verbose,
               bool out_to_file = false);

    ~LogMessage();

    std::ostream& stream() { return *stream_; }

  private:
    void init(const char* file, int line);

  private:
    std::ostream* stream_;
    Severity level_;
    bool verbose_;
    bool out_to_file_;

    static std::mutex lock_;  // stream write lock
    static std::string s_debug_logfile_;
    static std::string s_info_logfile_;
    static std::string s_warning_logfile_;
    static std::string s_error_logfile_;
    static std::string s_fatal_logfile_;

    DISALLOW_COPY_AND_ASSIGN(LogMessage);
};


}  // namespace log

}  // namespace ppspeech


#ifndef NDEBUG
#define DLOG_DEBUG \
    ppspeech::log::LogMessage(__FILE__, __LINE__, ppspeech::log::DEBUG, false)
#else
#define DLOG_DEBUG \
    ppspeech::log::LogMessage(__FILE__, __LINE__, ppspeech::log::DEBUG, true)
#endif

#define DLOG_INFO \
    ppspeech::log::LogMessage(__FILE__, __LINE__, ppspeech::log::INFO, true)
#define DLOG_WARNING \
    ppspeech::log::LogMessage(__FILE__, __LINE__, ppspeech::log::WARNING, true)
#define DLOG_ERROR \
    ppspeech::log::LogMessage(__FILE__, __LINE__, ppspeech::log::ERROR, true)
#define DLOG_FATAL \
    ppspeech::log::LogMessage(__FILE__, __LINE__, ppspeech::log::FATAL, true)

#define DLOG_0 DLOG_DEBUG
#define DLOG_1 DLOG_INFO
#define DLOG_2 DLOG_WARNING
#define DLOG_3 DLOG_ERROR
#define DLOG_4 DLOG_FATAL

#define LOG(level) DLOG_##level.stream()

#define VLOG(verboselevel) LOG(verboselevel)


#define CHECK(exp)                                        \
    ppspeech::log::LogMessage(                            \
        __FILE__, __LINE__, ppspeech::log::FATAL, !(exp)) \
            .stream()                                     \
        << "Check Failed: " #exp

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#ifndef NDEBUG
#define DCHECK(x) CHECK(x)
#define DCHECK_EQ(x, y) CHECK_EQ(x, y)
#define DCHECK_NE(x, y) CHECK_NE(x, y)
#define DCHECK_LE(x, y) CHECK_LE(x, y)
#define DCHECK_LT(x, y) CHECK_LT(x, y)
#define DCHECK_GE(x, y) CHECK_GE(x, y)
#define DCHECK_GT(x, y) CHECK_GT(x, y)
#else  // NDEBUG
#define DCHECK(condition) \
    while (false) CHECK(condition)
#define DCHECK_EQ(val1, val2) \
    while (false) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) \
    while (false) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) \
    while (false) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) \
    while (false) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) \
    while (false) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) \
    while (false) CHECK_GT(val1, val2)
#define DCHECK_STREQ(str1, str2) \
    while (false) CHECK_STREQ(str1, str2)
#endif