# Copyright (c) 2020 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = 'kaldi-util',
    srcs = [
       	'kaldi-holder.cc',
	      'kaldi-io.cc',
      	'kaldi-io-test.cc',
      	'kaldi-semaphore.cc',
      	'kaldi-table.cc',
      	'kaldi-thread.cc',
      	'parse-options.cc',
      	'simple-io-funcs.cc',
       	'simple-options.cc',
	      'text-utils.cc', ],
     hdrs = glob(["*.h"]),
     linkopts = ['-lpthread'],
     deps = [
		     '//base:kaldi-base',
         '//matrix:kaldi-matrix',
	       ],
)

cc_binary(
    name = 'kaldi-io-test',
    srcs = [
        'kaldi-io-test.cc',
        ],
    deps = [
      	':kaldi-util',
        ],
)

cc_binary(
    name = 'kaldi-table-test',
    srcs = [
        'kaldi-table-test.cc',
        ],
    deps = [
	      ':kaldi-util',
        ],
)
