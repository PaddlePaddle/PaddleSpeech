echo "Run decoder setup ..."

python decoder_setup.py install
rm -r ./build

echo "\nRun scorer setup ..."

python scorer_setup.py install
rm -r ./build

echo "\nFinish the installation of decoder and scorer."
