This directory contains data used in:

  Gorman, K., and Sproat, R. 2016. Minimally supervised number normalization.
  Transactions of the Association for Computational Linguistics 4: 507-519.

* `minimal.txt`: A list of 30 curated numbers used as the "minimal" training
  set.
* `random-trn.txt`: A list of 9000 randomly-generated numbers used as the
  "medium" training set.
* `random-tst.txt`: A list of 1000 randomly-generated numbers used as the test
  set.

Note that `random-trn.txt` and `random-tst.txt` are totally disjoint, but that
a small number of examples occur both in `minimal.txt` and `random-tst.txt`.

For information about the sampling procedure used to generate the random data
sets, see appendix A of the aforementioned paper.
