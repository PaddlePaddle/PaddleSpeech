#!/usr/bin/env python3
"""This script computes Word Error Rate and other related information.

Just given a reference and a hypothesis, the script closely matches
Kaldi's compute_wer binary.
Additionally, the script can produce human-readable edit distance
alignments, and find the top WER utterances and speakers.

Usage
-----

::

    Compute word error rate or a Levenshtein alignmentbetween a hypothesis and a reference.

    positional arguments:
      ref                   The ground truth to compare against. Text file with utterance-ID on the first column.
      hyp                   The hypothesis, for which WER is computed. Text file with utterance-ID on the first column.

    optional arguments:
      -h, --help            show this help message and exit
      --mode {present,all,strict}
                            How to treat missing hypotheses.
                             'present': only score hypotheses that were found
                             'all': treat missing hypotheses as empty
                             'strict': raise KeyError if a hypothesis is missing
      --print-top-wer       Print a list of utterances with the highest WER.
      --print-alignments    Print alignments for between all refs and hyps.Also has details for individual hyps. Outputs a lot of text.
      --align-separator ALIGN_SEPARATOR
                            When printing alignments, separate tokens with this.Note the spaces in the default.
      --align-empty ALIGN_EMPTY
                            When printing alignments, empty spaces are filled with this.
      --utt2spk UTT2SPK     Provide a mapping from utterance ids to speaker ids.If provided, print a list of speakers with highest WER.


Authors:
 * Aku Rouhe 2020
"""
import speechbrain.utils.edit_distance as edit_distance
import speechbrain.dataio.wer as wer_io


# These internal utilities read Kaldi-style text/utt2spk files:
def _plain_text_reader(path):
    # yields key, token_list
    with open(path, "r") as fi:
        for line in fi:
            key, *tokens = line.strip().split()
            yield key, tokens


def _plain_text_keydict(path):
    out_dict = {}  # key: token_list
    for key, tokens in _plain_text_reader(path):
        out_dict[key] = tokens
    return out_dict


def _utt2spk_keydict(path):
    utt2spk = {}
    with open(path, "r") as fi:
        for line in fi:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    return utt2spk


if __name__ == "__main__":
    import argparse

    # See: https://stackoverflow.com/a/22157136
    class SmartFormatter(argparse.HelpFormatter):
        def _split_lines(self, text, width):
            if text.startswith("R|"):
                return text[2:].splitlines()
            return argparse.HelpFormatter._split_lines(self, text, width)

    parser = argparse.ArgumentParser(
        description=(
            "Compute word error rate or a Levenshtein alignment"
            "between a hypothesis and a reference."
        ),
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        "ref",
        help="The ground truth to compare against. \
            Text file with utterance-ID on the first column.",
    )
    parser.add_argument(
        "hyp",
        help="The hypothesis, for which WER is computed. \
            Text file with utterance-ID on the first column.",
    )
    parser.add_argument(
        "--mode",
        help="R|How to treat missing hypotheses.\n"
        " 'present': only score hypotheses that were found\n"
        " 'all': treat missing hypotheses as empty\n"
        " 'strict': raise KeyError if a hypothesis is missing",
        choices=["present", "all", "strict"],
        default="strict",
    )
    parser.add_argument(
        "--print-top-wer",
        action="store_true",
        help="Print a list of utterances with the highest WER.",
    )
    parser.add_argument(
        "--print-alignments",
        action="store_true",
        help=(
            "Print alignments for between all refs and hyps."
            "Also has details for individual hyps. Outputs a lot of text."
        ),
    )
    parser.add_argument(
        "--align-separator",
        default=" ; ",
        help=(
            "When printing alignments, separate tokens with this."
            "Note the spaces in the default."
        ),
    )
    parser.add_argument(
        "--align-empty",
        default="<eps>",
        help="When printing alignments, empty spaces are filled with this.",
    )
    parser.add_argument(
        "--utt2spk",
        help="Provide a mapping from utterance ids to speaker ids."
        "If provided, print a list of speakers with highest WER.",
    )
    args = parser.parse_args()
    details_by_utterance = edit_distance.wer_details_by_utterance(
        _plain_text_keydict(args.ref),
        _plain_text_keydict(args.hyp),
        compute_alignments=args.print_alignments,
        scoring_mode=args.mode,
    )
    summary_details = edit_distance.wer_summary(details_by_utterance)
    wer_io.print_wer_summary(summary_details)
    if args.print_top_wer:
        top_non_empty, top_empty = edit_distance.top_wer_utts(
            details_by_utterance
        )
        wer_io._print_top_wer_utts(top_non_empty, top_empty)
    if args.utt2spk:
        utt2spk = _utt2spk_keydict(args.utt2spk)
        details_by_speaker = edit_distance.wer_details_by_speaker(
            details_by_utterance, utt2spk
        )
        top_spks = edit_distance.top_wer_spks(details_by_speaker)
        wer_io._print_top_wer_spks(top_spks)
    if args.print_alignments:
        wer_io.print_alignments(
            details_by_utterance,
            empty_symbol=args.align_empty,
            separator=args.align_separator,
        )
