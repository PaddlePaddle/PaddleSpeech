# for plain text
python3 cn_tn.py example_plain.txt output_plain.txt
diff example_plain.txt output_plain.txt

# for Kaldi's trans format
python3 cn_tn.py --has_key example_kaldi.txt output_kaldi.txt
diff example_kaldi.txt output_kaldi.txt

