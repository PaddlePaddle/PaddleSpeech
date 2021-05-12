.PHONY: help
help:
	@echo "merge_unihan          merge Unihan data"
	@echo "pua                   generate PUA"
	@echo "check                 check unexpected char"

.PHONY: merge_unihan
merge_unihan: check
	python merge_unihan.py

.PHONY: pua
pua:
	python tools/gen_gb_pua.py > GBK_PUA.txt

.PHONY: check
check:
	-rg 'ɡ|ɑ|í|è'
