#include "base/log.h"

DEFINE_int32(logtostderr, 0, "logging to stderr");

namespace ppspeech {

static char __progname[] = "paddlespeech";

namespace log {

std::mutex LogMessage::lock_;
std::string LogMessage::s_debug_logfile_("");
std::string LogMessage::s_info_logfile_("");
std::string LogMessage::s_warning_logfile_("");
std::string LogMessage::s_error_logfile_("");
std::string LogMessage::s_fatal_logfile_("");

void LogMessage::get_curr_proc_info(std::string* pid, std::string* proc_name) {
    std::stringstream ss;
    ss << getpid();
    ss >> *pid;
    *proc_name = ::ppspeech::__progname;
}

LogMessage::LogMessage(const char* file,
                       int line,
                       Severity level,
                       bool verbose,
                       bool out_to_file /* = false */)
    : level_(level), verbose_(verbose), out_to_file_(out_to_file) {
    if (FLAGS_logtostderr == 0) {
        stream_ = std::shared_ptr<std::ostream>(&std::cout);
    } else if (FLAGS_logtostderr == 1) {
        stream_ = std::shared_ptr<std::ostream>(&std::cerr);
    } else if (out_to_file_) {
        // logfile
        lock_.lock();
        init(file, line);
    }
}

LogMessage::~LogMessage() {
    stream() << std::endl;

    if (out_to_file_) {
        lock_.unlock();
    }

    if (level_ == FATAL) {
        std::abort();
    }
}

void LogMessage::init(const char* file, int line) {
    time_t t = time(0);
    char tmp[100];
    strftime(tmp, sizeof(tmp), "%Y%m%d-%H%M%S", localtime(&t));

    if (s_info_logfile_.empty()) {
        std::string pid;
        std::string proc_name;
        get_curr_proc_info(&pid, &proc_name);

        s_debug_logfile_ =
            std::string("log." + proc_name + ".log.DEBUG." + tmp + "." + pid);
        s_info_logfile_ =
            std::string("log." + proc_name + ".log.INFO." + tmp + "." + pid);
        s_warning_logfile_ =
            std::string("log." + proc_name + ".log.WARNING." + tmp + "." + pid);
        s_error_logfile_ =
            std::string("log." + proc_name + ".log.ERROR." + tmp + "." + pid);
        s_fatal_logfile_ =
            std::string("log." + proc_name + ".log.FATAL." + tmp + "." + pid);
    }

    std::ofstream ofs;
    if (level_ == DEBUG) {
        stream_ = std::make_shared<std::ofstream>(
            s_debug_logfile_.c_str(), std::ios::out | std::ios::app);
        // ofs.open(s_debug_logfile_.c_str(), std::ios::out | std::ios::app);
    } else if (level_ == INFO) {
        // ofs.open(s_warning_logfile_.c_str(), std::ios::out | std::ios::app);
        stream_ = std::make_shared<std::ofstream>(
            s_warning_logfile_.c_str(), std::ios::out | std::ios::app);
    } else if (level_ == WARNING) {
        // ofs.open(s_warning_logfile_.c_str(), std::ios::out | std::ios::app);
        stream_ = std::make_shared<std::ofstream>(
            s_warning_logfile_.c_str(), std::ios::out | std::ios::app);
    } else if (level_ == ERROR) {
        // ofs.open(s_error_logfile_.c_str(), std::ios::out | std::ios::app);
        stream_ = std::make_shared<std::ofstream>(
            s_error_logfile_.c_str(), std::ios::out | std::ios::app);
    } else {
        // ofs.open(s_fatal_logfile_.c_str(), std::ios::out | std::ios::app);
        stream_ = std::make_shared<std::ofstream>(
            s_fatal_logfile_.c_str(), std::ios::out | std::ios::app);
    }

    // stream_ = &ofs;

    stream() << tmp << " " << file << " line " << line << "; ";
    stream() << std::flush;
}
}  // namespace log
}  // namespace ppspeech