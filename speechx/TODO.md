# TODO

* DecibelNormalizer: there is a little bit difference between offline and online db norm. The computation of online db norm read feature chunk by chunk, which causes the feature size is different with offline db norm. In normalizer.cc:73, the samples.size() is different, which causes the difference of result.
