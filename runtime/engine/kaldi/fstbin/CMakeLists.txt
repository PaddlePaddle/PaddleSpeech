cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(BINS
fstaddselfloops
fstisstochastic    
fstminimizeencoded
fstdeterminizestar
fsttablecompose
)

foreach(binary IN LISTS BINS)
   add_executable(${binary} ${CMAKE_CURRENT_SOURCE_DIR}/${binary}.cc) 
   target_include_directories(${binary} PRIVATE ${SPEECHX_ROOT} ${SPEECHX_ROOT}/kaldi)
   target_link_libraries(${binary} PUBLIC kaldi-fstext glog gflags fst dl)
endforeach()
