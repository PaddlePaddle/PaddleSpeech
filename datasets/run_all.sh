cd librispeech
python librispeech.py
if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi
cd -

cat librispeech/manifest.train* | shuf > manifest.train
cat librispeech/manifest.dev-clean > manifest.dev
cat librispeech/manifest.test-clean > manifest.test

echo "All done."
