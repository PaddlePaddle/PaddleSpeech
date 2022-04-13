// fstbin/fsttablecompose.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "fstext/table-matcher.h"
#include "util/parse-options.h"

/*
  cd ~/tmpdir
  while true; do
    fstrand  | fstarcsort --sort_type=olabel > 1.fst; fstrand | fstarcsort
  > 2.fst fstcompose 1.fst 2.fst > 3a.fst fsttablecompose 1.fst 2.fst > 3b.fst
    fstequivalent --random=true 3a.fst 3b.fst || echo "Test failed"
    echo -n "."
  done

*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;  // NOLINT
    using namespace fst;  // NOLINT
    using kaldi::int32;
    /*
      fsttablecompose should always give equivalent results to compose,
      but it is more efficient for certain kinds of inputs.
      In particular, it is useful when, say, the left FST has states
      that typically either have epsilon olabels, or
      one transition out for each of the possible symbols (as the
      olabel).  The same with the input symbols of the right-hand FST
      is possible.
    */

    const char *usage =
        "Composition algorithm [between two FSTs of standard type, in "
        "tropical\n"
        "semiring] that is more efficient for certain cases-- in particular,\n"
        "where one of the FSTs (the left one, if --match-side=left) has large\n"
        "out-degree\n"
        "\n"
        "Usage:  fsttablecompose (fst1-rxfilename|fst1-rspecifier) "
        "(fst2-rxfilename|fst2-rspecifier) [(out-rxfilename|out-rspecifier)]\n";

    ParseOptions po(usage);

    TableComposeOptions opts;
    std::string match_side = "left";
    std::string compose_filter = "sequence";

    po.Register("connect", &opts.connect, "If true, trim FST before output.");
    po.Register("match-side", &match_side,
                "Side of composition to do table "
                "match, one of: \"left\" or \"right\".");
    po.Register("compose-filter", &compose_filter,
                "Composition filter to use, "
                "one of: \"alt_sequence\", \"auto\", \"match\", \"sequence\"");

    po.Read(argc, argv);

    if (match_side == "left") {
      opts.table_match_type = MATCH_OUTPUT;
    } else if (match_side == "right") {
      opts.table_match_type = MATCH_INPUT;
    } else {
      KALDI_ERR << "Invalid match-side option: " << match_side;
    }

    if (compose_filter == "alt_sequence") {
      opts.filter_type = ALT_SEQUENCE_FILTER;
    } else if (compose_filter == "auto") {
      opts.filter_type = AUTO_FILTER;
    } else if (compose_filter == "match") {
      opts.filter_type = MATCH_FILTER;
    } else if (compose_filter == "sequence") {
      opts.filter_type = SEQUENCE_FILTER;
    } else {
      KALDI_ERR << "Invalid compose-filter option: " << compose_filter;
    }

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst1_in_str = po.GetArg(1), fst2_in_str = po.GetArg(2),
                fst_out_str = po.GetOptArg(3);

    VectorFst<StdArc> *fst1 = ReadFstKaldi(fst1_in_str);

    VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);

    // Checks if <fst1> is olabel sorted and <fst2> is ilabel sorted.
    if (fst1->Properties(fst::kOLabelSorted, true) == 0) {
      KALDI_WARN << "The first FST is not olabel sorted.";
    }
    if (fst2->Properties(fst::kILabelSorted, true) == 0) {
      KALDI_WARN << "The second FST is not ilabel sorted.";
    }

    VectorFst<StdArc> composed_fst;

    TableCompose(*fst1, *fst2, &composed_fst, opts);

    delete fst1;
    delete fst2;

    WriteFstKaldi(composed_fst, fst_out_str);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
