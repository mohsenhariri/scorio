
SRC:=scorio/
JULIA_PROJECT:=julia/Scorio.jl

.PHONY: format format-check lint clean build install test pkg-check pkg-publish-test pkg-publish docs docs-clean help julia-test julia-docs julia-docs-clean julia-install test-comparison

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

landing-serve:
	python -m http.server --directory docs-landing 4002

# Julia commands
julia-install:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'
	julia --project=$(JULIA_PROJECT)/docs -e 'using Pkg; Pkg.develop(path="$(JULIA_PROJECT)"); Pkg.instantiate()'

julia-test:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.test()'

julia-docs:
	julia --project=$(JULIA_PROJECT)/docs $(JULIA_PROJECT)/docs/make.jl

julia-docs-clean:
	rm -rf $(JULIA_PROJECT)/docs/build

julia-docs-serve:
	python -m http.server --directory $(JULIA_PROJECT)/docs/build 4001

julia-clean: julia-docs-clean
