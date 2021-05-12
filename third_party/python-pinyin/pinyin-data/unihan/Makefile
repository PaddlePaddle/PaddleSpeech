.PHONY: help
help:
	@echo "parse		parse Unihan database "
	@echo "update		update Unihan database"
	@echo "diff		diff between Unihan data and parsed data"

.PHONY:parse
parse:
	@python parse_pinyin.py

.PHONY:update
update:
	-rm Unihan*
	wget ftp://ftp.unicode.org/Public/UNIDATA/Unihan.zip -O Unihan.zip
	unzip Unihan.zip
	python parse_pinyin.py

.PHONY:diff
diff:
	@bash diff.sh
