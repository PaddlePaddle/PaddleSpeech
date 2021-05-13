PYTHON:= python3.8
.PHONY: all clean

all: virtualenv

virtualenv:
	test -d venv || virtualenv -p $(PYTHON) venv
	touch venv/bin/activate

clean:
	rm -fr venv
	find -iname "*.pyc" -delete

