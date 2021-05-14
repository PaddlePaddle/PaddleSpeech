# Text normalization covering grammars

This repository provides covering grammars for English and Russian text normalization as
documented in:
  
  Gorman, K., and Sproat, R. 2016. Minimally supervised number normalization.
  _Transactions of the Association for Computational Linguistics_ 4: 507-519.
  
  Ng, A. H., Gorman, K., and Sproat, R. 2017. Minimally supervised
  written-to-spoken text normalization. In _ASRU_, pages 665-670.

If you use these grammars in a publication, we would appreciate if you cite these works.

## Building

The grammars are written in [Thrax](thrax.opengrm.org) and compile into [OpenFst](openfst.org) FAR (FstARchive) files. To compile, simply run `make` in the `src/` directory.

## License

See `LICENSE`.

## Mandatory disclaimer

This is not an official Google product.
