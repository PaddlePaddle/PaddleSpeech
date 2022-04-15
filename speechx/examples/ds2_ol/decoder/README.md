# ASR Decoder

ASR Decoder test bins. We using theses bins to test CTC BeamSearch decoder and WFST decoder.

* decoder_test_main.cc 
feed nnet output logprob, and only test decoder

* offline_decoder_sliding_chunk_main.cc
feed streaming audio feature, decode as streaming manner.

* offline_wfst_decoder_main.cc
feed streaming audio feature, decode using WFST as streaming manner.