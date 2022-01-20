# Speaker Diarization on AMI corpus

## About the AMI corpus:
"The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals synchronized to a common timeline. These include close-talking and far-field microphones, individual and room-view video cameras, and output from a slide projector and an electronic whiteboard. During the meetings, the participants also have unsynchronized pens available to them that record what is written. The meetings were recorded in English using three different rooms with different acoustic properties, and include mostly non-native speakers." See [ami overview](http://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) for more details.

## About the example
The script performs diarization using x-vectors(TDNN,ECAPA-TDNN) on the AMI mix-headset data. We demonstrate the use of different clustering methods: AHC, spectral.

## How to Run
Use the following command to run diarization on AMI corpus.
`bash ./run.sh` 

## Results (DER) coming soon! :)