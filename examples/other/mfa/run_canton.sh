exp=exp

mkdir -p $exp
LEXICON_NAME='canton'
MFA_DOWNLOAD_DIR=local/

if [ ! -f "$exp/$LEXICON_NAME.lexicon" ]; then
    echo "generating lexicon and training data..."
    python local/generate_canton_lexicon_wavlabs.py --output_lexicon "$exp/$LEXICON_NAME.lexicon" --output_wavlabs "$exp/$LEXICON_NAME"_wavlabs --inputs ~/datasets/Guangzhou_Cantonese_Scripted_Speech_Corpus_Daily_Use_Sentence ~/datasets/Guangzhou_Cantonese_Scripted_Speech_Corpus_in_Vehicle
    echo "lexicon and training data done"
fi

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
if [ ! -d "$exp/canton_alignment" ]; then
    echo "Start MFA training..."
    PATH=$MFA_DOWNLOAD_DIR/montreal-forced-aligner/bin/:$PATH \
    LD_LIBRARY_PATH=$MFA_DOWNLOAD_DIR/montreal-forced-aligner/lib/:$LD_LIBRARY_PATH \
    ./$MFA_DOWNLOAD_DIR/montreal-forced-aligner/bin/mfa_train_and_align \
        "$exp/$LEXICON_NAME"_wavlabs "$exp/$LEXICON_NAME.lexicon" $exp/canton_alignment -o $exp/canton_model --clean --verbose -j 10 --temp_directory $exp/.mfa_train_and_align
    echo "training done!"
    echo "results: $exp/canton_alignment"
    echo "model: $exp/canton_model"
fi

