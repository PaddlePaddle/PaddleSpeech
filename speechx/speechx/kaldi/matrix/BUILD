# Copyright (c) 2020 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = 'kaldi-matrix',
    srcs = [
       	'compressed-matrix.cc',
				'kaldi-matrix.cc',
				'kaldi-vector.cc',
				'matrix-functions.cc',
				'optimization.cc',
				'packed-matrix.cc',
				'qr.cc',
				'sparse-matrix.cc',
				'sp-matrix.cc',
				'srfft.cc',
				'tp-matrix.cc',
	  ],
    hdrs = glob(["*.h"]),
    deps = [
		    '//base:kaldi-base',
        '//common/third_party/openblas:openblas',
    ],
    linkopts=['-lgfortran'],
)

cc_binary(
    name = 'matrix-lib-test',
    srcs = [
        'matrix-lib-test.cc',
        ],
    deps = [
       	':kaldi-matrix',
        '//util:kaldi-util',
        ],
)

