# Use Montreal-Forced-Aligner
Here is an example to use [MFA1.x](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).
Run the following script to get started, for more detail, please see `run.sh`.
```bash
./run.sh
```
# Rhythm tags for MFA
If you want to get rhythm tags with duration through MFA tool, you may add flag `--rhy-with-duration` in the first two commands in `run.sh`
Note that only CSMSC dataset is supported so far, and we replace `#` with `sp` in rhythm tags for MFA.
