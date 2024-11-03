.PHONY: clean install test lint build publish-test publish

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install:
	pip install -e .

test:
	pytest

lint:
	pip install flake8
	flake8 simple_math tests

build: clean
	python setup.py sdist bdist_wheel

publish-test: build
	pip install twine
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish: build
	pip install twine
	twine upload dist/*

venv:
	python -m venv venv
	@echo "Now run 'source venv/bin/activate' to activate the virtual environment"

dev-install:
	pip install -e ".[dev]"
	pip install flake8 wheel twine

help:
	@echo "make clean      - Remove all build, test, coverage and Python artifacts"
	@echo "make install    - Install the package in development mode"
	@echo "make test       - Run unit tests"
	@echo "make lint       - Check style with flake8"
	@echo "make build      - Build source and wheel package"
	@echo "make publish-test - Upload package to TestPyPI"
	@echo "make publish    - Upload package to PyPI"
	@echo "make venv      - Create a new virtual environment"
	@echo "make dev-install - Install development dependencies"