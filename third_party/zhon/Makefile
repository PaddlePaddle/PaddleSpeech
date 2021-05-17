PROJECT = zhon

.PHONY: docs clean lint test test-all coverage dist release

help:
	@echo "clean - remove all build artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the current Python"
	@echo "test-all - run tests in all environments"
	@echo "coverage - check code coverage"
	@echo "docs - generate Sphinx HTML documentation"
	@echo "dist - make the source and binary distributions"
	@echo "release - package and upload a release"

clean:
	rm -rf build dist egg *.egg-info htmlcov
	find . -name '*.py[co]' -exec rm -f {} +
	$(MAKE) -C docs clean

lint:
	flake8 $(PROJECT) tests setup.py

test:
	python setup.py test

test-all:
	tox

coverage:
	coverage run --source $(PROJECT) setup.py test
	coverage report --fail-under=100

docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html

dist: clean
	python setup.py sdist bdist_wheel

release: clean dist
	twine upload -s dist/*
