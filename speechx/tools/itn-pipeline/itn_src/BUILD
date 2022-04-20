package(default_visibility = ['//visibility:public'])

cc_binary(
  name = "itn_main",
	srcs = ["itn_main.cc"],
	deps = [
	    ":inverse_text_normalizer_impl",
	],
)

cc_library(
  name = "string_composer",
	srcs = ["string_composer.cc"],
	hdrs = ["string_composer.h"],
	deps = [
	    "//third_party/openfst:fst",
			"//base:compat",
	],
)

cc_library(
  name = "blank_processor",
	srcs = ["blank_processor.cc"],
	hdrs = ["blank_processor.h"],
	deps = [
	    "//base:compat",
  ],
)

cc_library(
  name = "inverse_text_normalizer_impl",
  srcs = ["inverse_text_normalizer_impl.cc"],
	hdrs = ["inverse_text_normalizer_impl.h"],
	deps = [
	    ":string_composer",
      ":blank_processor",
	    "//base:compat",
  ],
)
