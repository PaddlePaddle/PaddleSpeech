exp=exp
data=data

mkdir -p $exp
mkdir -p $data

LEXICON_NAME='simple'
MFA_DOWNLOAD_DIR=local/

if [ ! -f "$exp/$LEXICON_NAME.lexicon" ]; then
    echo "generating lexicon..."
    python local/generate_lexicon.py "$exp/$LEXICON_NAME" --with-r --with-tone
    echo "lexicon done"
fi

if [ ! -d $exp/baker_corpus ]; then
    echo "reorganizing baker corpus..."
    python local/reorganize_baker.py --root-dir=~/datasets/BZNSYP --output-dir=$exp/baker_corpus --resample-audio
    echo "reorganization done. Check output in $exp/baker_corpus."
    echo "audio files are resampled to 16kHz"
    echo "transcription for each audio file is saved with the same namd in $exp/baker_corpus "
fi


echo "detecting oov..."
python local/detect_oov.py $exp/baker_corpus $exp/"$LEXICON_NAME.lexicon"
echo "detecting oov done. you may consider regenerate lexicon if there is unexpected OOVs."


if [ ! -f "$MFA_DOWNLOAD_DIR/montreal-forced-aligner_linux.tar.gz" ]; then
    echo "downloading mfa..."
    (cd $MFA_DOWNLOAD_DIR && wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz)
    echo "download mfa done!"
fi

if [ ! -d "$MFA_DOWNLOAD_DIR/montreal-forced-aligner" ]; then
    echo "extracting mfa..."
    (cd $MFA_DOWNLOAD_DIR && tar xvf "montreal-forced-aligner_linux.tar.gz")
    echo "extraction done!"
fi

export PATH="$MFA_DOWNLOAD_DIR/montreal-forced-aligner/bin"

if [ ! -d "$exp/baker_alignment" ]; then
    echo "Start MFA training..."
    PATH=$MFA_DOWNLOAD_DIR/montreal-forced-aligner/bin/:$PATH \
    LD_LIBRARY_PATH=$MFA_DOWNLOAD_DIR/montreal-forced-aligner/lib/:$LD_LIBRARY_PATH \
    ./$MFA_DOWNLOAD_DIR/montreal-forced-aligner/bin/mfa_train_and_align \
        $exp/baker_corpus "$exp/$LEXICON_NAME.lexicon" $exp/baker_alignment -o $exp/baker_model --clean --verbose -j 10 --temp_directory $exp/.mfa_train_and_align
    echo "training done!"
    echo "results: $exp/baker_alignment"
    echo "model: $exp/baker_model"
fi

