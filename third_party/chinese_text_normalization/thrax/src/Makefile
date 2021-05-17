en/verbalizer/podspeech.far: en/verbalizer/podspeech.grm util/util.far util/case.far en/verbalizer/extra_numbers.far en/verbalizer/float.far en/verbalizer/math.far en/verbalizer/miscellaneous.far en/verbalizer/money.far en/verbalizer/numbers.far en/verbalizer/numbers_plus.far en/verbalizer/spelled.far en/verbalizer/spoken_punct.far en/verbalizer/time.far en/verbalizer/urls.far
	thraxcompiler --input_grammar=$< --output_far=$@

util/util.far: util/util.grm util/byte.far util/case.far
	thraxcompiler --input_grammar=$< --output_far=$@

util/byte.far: util/byte.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

util/case.far: util/case.grm util/byte.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/extra_numbers.far: en/verbalizer/extra_numbers.grm util/byte.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/numbers.far: en/verbalizer/numbers.grm en/verbalizer/number_names.far util/byte.far universal/thousands_punct.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/number_names.far: en/verbalizer/number_names.grm util/arithmetic.far en/verbalizer/g.fst en/verbalizer/cardinals.tsv en/verbalizer/ordinals.tsv
	thraxcompiler --input_grammar=$< --output_far=$@

util/arithmetic.far: util/arithmetic.grm util/byte.far util/germanic.tsv
	thraxcompiler --input_grammar=$< --output_far=$@

universal/thousands_punct.far: universal/thousands_punct.grm util/byte.far util/util.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/float.far: en/verbalizer/float.grm en/verbalizer/factorization.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/factorization.far: en/verbalizer/factorization.grm util/byte.far util/util.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/lexical_map.far: en/verbalizer/lexical_map.grm util/byte.far en/verbalizer/lexical_map.tsv
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/math.far: en/verbalizer/math.grm en/verbalizer/float.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/miscellaneous.far: en/verbalizer/miscellaneous.grm util/byte.far ru/classifier/cyrillic.far en/verbalizer/extra_numbers.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far en/verbalizer/spelled.far
	thraxcompiler --input_grammar=$< --output_far=$@

ru/classifier/cyrillic.far: ru/classifier/cyrillic.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/spelled.far: en/verbalizer/spelled.grm util/byte.far ru/classifier/cyrillic.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/money.far: en/verbalizer/money.grm util/byte.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far en/verbalizer/money.tsv
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/numbers_plus.far: en/verbalizer/numbers_plus.grm en/verbalizer/factorization.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/spoken_punct.far: en/verbalizer/spoken_punct.grm en/verbalizer/lexical_map.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/time.far: en/verbalizer/time.grm util/byte.far en/verbalizer/lexical_map.far en/verbalizer/numbers.far
	thraxcompiler --input_grammar=$< --output_far=$@

en/verbalizer/urls.far: en/verbalizer/urls.grm util/byte.far en/verbalizer/lexical_map.far
	thraxcompiler --input_grammar=$< --output_far=$@

clean:
	rm -f util/util.far util/case.far en/verbalizer/extra_numbers.far en/verbalizer/float.far en/verbalizer/math.far en/verbalizer/miscellaneous.far en/verbalizer/money.far en/verbalizer/numbers.far en/verbalizer/numbers_plus.far en/verbalizer/spelled.far en/verbalizer/spoken_punct.far en/verbalizer/time.far en/verbalizer/urls.far util/byte.far en/verbalizer/number_names.far universal/thousands_punct.far util/arithmetic.far en/verbalizer/factorization.far en/verbalizer/lexical_map.far ru/classifier/cyrillic.far
