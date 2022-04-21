PATH := $(PATH):../tools/fc_base/thrax-build/src/bin:../tools/fc_base/openfst-build/src/bin/:../tools/fc_base/openfst-build/src/extensions/far/
LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../tools/fc_base/thrax-build/src/lib


build:
	mkdir -p build

# common

common/byte.far: common/byte.grm
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

common/util.far: common/util.grm common/byte.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

# tagger

taggers/int.far: taggers/int.grm common/byte.far common/util.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/float.far: taggers/float.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/time.far: taggers/time.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/measure.far: taggers/measure.grm common/byte.far common/util.far taggers/int.far taggers/float.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/telephone.far: taggers/telephone.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/electronic.far: taggers/electronic.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/fraction.far: taggers/fraction.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/percentage.far: taggers/percentage.grm common/byte.far common/util.far taggers/int.far taggers/float.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/date.far: taggers/date.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/fraction_structured.far: taggers/fraction_structured.grm common/byte.far common/util.far taggers/int.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/word_structured.far: taggers/word_structured.grm common/byte.far common/util.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/whitelist.far: taggers/whitelist.grm common/byte.far common/util.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/replace_num.far: taggers/replace_num.grm common/byte.far common/util.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

taggers/rule_fst.far: taggers/rule.grm common/byte.far common/util.far taggers/int.far taggers/float.far taggers/time.far taggers/measure.far taggers/electronic.far taggers/fraction_structured.far taggers/word_structured.far taggers/whitelist.far taggers/percentage.far taggers/fraction.far taggers/date.far taggers/telephone.far
	thraxcompiler --save_symbols --input_grammar=$< --output_far=$@

build/rule_fst: build taggers/rule_fst.far
	farextract --filename_suffix=".fst" --filename_prefix="build/" taggers/rule_fst.far
	touch build/done_rule_fst


build/replace_fst: build taggers/replace_num.far
	farextract --filename_suffix=".fst" --filename_prefix="build/" taggers/replace_num.far
	touch build/done_repalce_num

move_far_to_build_dir: build
	mv common/*.far build
	mv taggers/*.far build
	mv verbalizers/*.far build

#all: build/rule_fst build/replace_fst
all: build/replace_fst

.PHONY: clean move_far_to_build_dir

clean:
	rm -rf build common/*.far taggers/*.far verbalizers/*.far
