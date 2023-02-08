EXP_DIR=exp

mkdir -p $EXP_DIR
LEXICON_NAME='canton'
if [ ! -f "$EXP_DIR/$LEXICON_NAME.lexicon" ]; then
    echo "generating lexicon and training data..."
    python local/generate_canton_lexicon_wavlabs.py --output_lexicon "$EXP_DIR/$LEXICON_NAME.lexicon" --output_wavlabs "$EXP_DIR/$LEXICON_NAME"_wavlabs --inputs ~/dataset/Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence ~/dataset/Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle
    echo "lexicon and training data done"
fi


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
if [ ! -d "$EXP_DIR/canton_alignment" ]; then
    echo "Start MFA training..."
    mfa_train_and_align "$EXP_DIR/$LEXICON_NAME"_wavlabs "$EXP_DIR/$LEXICON_NAME.lexicon" $EXP_DIR/canton_alignment -o $EXP_DIR/canton_model --clean --verbose --temp_directory $EXP_DIR/.mfa_train_and_align
    echo "training done!"
    echo "results: $EXP_DIR/canton_alignment"
    echo "model: $EXP_DIR/canton_model"
fi

