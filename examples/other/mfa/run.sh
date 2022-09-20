EXP_DIR=exp

mkdir -p $EXP_DIR
LEXICON_NAME='simple'
if [ ! -f "$EXP_DIR/$LEXICON_NAME.lexicon" ]; then
    echo "generating lexicon..."
    python local/generate_lexicon.py "$EXP_DIR/$LEXICON_NAME" --with-r --with-tone
    echo "lexicon done"
fi

if [ ! -d $EXP_DIR/baker_corpus ]; then
    echo "reorganizing baker corpus..."
    python local/reorganize_baker.py --root-dir=~/datasets/BZNSYP --output-dir=$EXP_DIR/baker_corpus --resample-audio
    echo "reorganization done. Check output in $EXP_DIR/baker_corpus."
    echo "audio files are resampled to 16kHz"
    echo "transcription for each audio file is saved with the same namd in $EXP_DIR/baker_corpus "
fi


echo "detecting oov..."
python local/detect_oov.py $EXP_DIR/baker_corpus $EXP_DIR/"$LEXICON_NAME.lexicon"
echo "detecting oov done. you may consider regenerate lexicon if there is unexpected OOVs."


MFA_DOWNLOAD_DIR=local/

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
if [ ! -d "$EXP_DIR/baker_alignment" ]; then
    echo "Start MFA training..."
    mfa_train_and_align $EXP_DIR/baker_corpus "$EXP_DIR/$LEXICON_NAME.lexicon" $EXP_DIR/baker_alignment -o $EXP_DIR/baker_model --clean --verbose --temp_directory $EXP_DIR/.mfa_train_and_align
    echo "training done!"
    echo "results: $EXP_DIR/baker_alignment"
    echo "model: $EXP_DIR/baker_model"
fi

