#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # remove punctuation except apostrophe
  s/<space>/spacemark/g;  # for scoring
  s/'/apostrophe/g;
  s/[[:punct:]]//g;
  s/apostrophe/'/g;
  s/spacemark/<space>/g;  # for scoring

  # remove whitespace
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
