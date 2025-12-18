
SRC:=scorio/

.PHONY: format format-check lint clean build install test pkg-check pkg-publish-test pkg-publish docs docs-clean help

format-check:
	isort --check-only $(SRC)
	black --check $(SRC)

format:
	isort $(SRC)
	black $(SRC)

lint:
	mypy $(SRC)

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	pip install --upgrade build
	python -m build

install:
	pip install -e ".[dev]"

test:
	pytest test/

pkg-check: build
	python -m pip install --upgrade twine
	twine check dist/*

pkg-publish-test: pkg-check
	@echo "Publishing to TestPyPI..."
	twine upload -r testpypi dist/* --verbose

pkg-publish: pkg-check
	twine upload dist/* --verbose

docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

docs-serve:
	python -m http.server --directory docs/_build/html 4000