echo "Run decoder setup ..."

python decoder_setup.py install
rm -r ./build

echo "Run scorer setup ..."

python scorer_setup.py install
rm -r ./build

echo "Finish the installation of decoder and scorer."
