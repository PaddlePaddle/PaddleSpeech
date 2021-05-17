itn.far: itn.grm byte.far number.far hotfix.far percentage.far date.far amount.far
	thraxcompiler --input_grammar=$< --output_far=$@

byte.far: byte.grm 
	thraxcompiler --input_grammar=$< --output_far=$@

number.far: number.grm byte.far
	thraxcompiler --input_grammar=$< --output_far=$@

hotfix.far: hotfix.grm byte.far hotfix.list
	thraxcompiler --input_grammar=$< --output_far=$@

percentage.far: percentage.grm byte.far number.far
	thraxcompiler --input_grammar=$< --output_far=$@

date.far: date.grm byte.far number.far
	thraxcompiler --input_grammar=$< --output_far=$@

amount.far: amount.grm byte.far number.far
	thraxcompiler --input_grammar=$< --output_far=$@

clean:
	rm -f byte.far number.far hotfix.far percentage.far date.far amount.far
