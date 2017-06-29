%module swig_ctc_beam_search_decoder
%{
#include "ctc_beam_search_decoder.h"
%}

%include "std_vector.i"
%include "std_pair.i"
%include "std_string.i"

namespace std{
    %template(DoubleVector) std::vector<double>;
    %template(IntVector) std::vector<int>;
    %template(StringVector) std::vector<std::string>;
    %template(VectorOfStructVector) std::vector<std::vector<double> >;
    %template(FloatVector) std::vector<float>;
    %template(Pair) std::pair<float, std::string>;
    %template(PairFloatStringVector)  std::vector<std::pair<float, std::string> >;
    %template(PairDoubleStringVector) std::vector<std::pair<double, std::string> >;
}

%import scorer.h
%include "ctc_beam_search_decoder.h"
