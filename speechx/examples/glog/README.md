# [GLOG](https://rpg.ifi.uzh.ch/docs/glog.html)

Unless otherwise specified, glog writes to the filename `/tmp/<program name>.<hostname>.<user name>.log.<severity level>.<date>.<time>.<pid>` (e.g., "/tmp/hello_world.example.com.hamaji.log.INFO.20080709-222411.10474"). By default, glog copies the log messages of severity level ERROR or FATAL to standard error (stderr) in addition to log files.

Several flags influence glog's output behavior. If the Google gflags library is installed on your machine, the configure script (see the INSTALL file in the package for detail of this script) will automatically detect and use it, allowing you to pass flags on the command line. For example, if you want to turn the flag --logtostderr on, you can start your application with the following command line:

   `./your_application --logtostderr=1`

If the Google gflags library isn't installed, you set flags via environment variables, prefixing the flag name with "GLOG_", e.g.

   `GLOG_logtostderr=1 ./your_application`

You can also modify flag values in your program by modifying global variables `FLAGS_*` . Most settings start working immediately after you update `FLAGS_*` . The exceptions are the flags related to destination files. For example, you might want to set `FLAGS_log_dir` before calling `google::InitGoogleLogging` . Here is an example:
∂∂
```c++
   LOG(INFO) << "file";
   // Most flags work immediately after updating values.
   FLAGS_logtostderr = 1;
   LOG(INFO) << "stderr";
   FLAGS_logtostderr = 0;
   // This won't change the log destination. If you want to set this
   // value, you should do this before google::InitGoogleLogging .
   FLAGS_log_dir = "/some/log/directory";
   LOG(INFO) << "the same file";
```
