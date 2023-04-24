include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_library(kaldi-native-fbank-core
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  log.cc
  mel-computations.cc
  rfft.cc
)
# We are using std::call_once() in log.h,which requires us to link with -pthread
if(NOT WIN32)
  target_link_libraries(kaldi-native-fbank-core -pthread)
endif()

if(KNF_HAVE_EXECINFO_H)
  target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_EXECINFO_H=1)
endif()

if(KNF_HAVE_CXXABI_H)
  target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_CXXABI_H=1)
endif()
