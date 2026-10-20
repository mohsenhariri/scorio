
SRC:=scorio/
JULIA_PROJECT:=julia/Scorio.jl

.PHONY: format format-check lint clean build install test pkg-check pkg-publish-test pkg-publish sync-version release-py release-jl release-js js-build js-test jl-install jl-test jl-test-slow py-docs-build py-docs-clean py-docs-serve jl-docs-build jl-docs-clean jl-docs-serve js-docs-build js-docs-check js-docs-clean js-docs-serve landing-serve
.PHONY: test-eval-py test-rank-py test-rank-py-slow test-eval-jl test-rank-jl test-rank-jl-slow
.PHONY: docs-site-build docs-seo-check docs-seo-test

format-check:
	ruff format --check $(SRC)
	ruff check $(SRC)

format:
	ruff format $(SRC)
	ruff check --fix $(SRC)

lint:
	ruff check $(SRC)
	mypy $(SRC)

sync-version:
	python scripts/sync_version.py

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
	pytest tests/

test-eval-py:
	pytest tests/eval

test-rank-py:
	pytest tests/rank -m "not slow"

test-rank-py-slow:
	pytest tests/rank -m slow

pkg-check: build
	uv pip install --upgrade twine
	twine check dist/*

pkg-publish-test: pkg-check
	@echo "Publishing to TestPyPI..."
	twine upload -r testpypi dist/* --verbose

pkg-publish: pkg-check
	twine upload dist/* --verbose

release-py:
	./scripts/release_github.sh py

release-jl:
	./scripts/release_github.sh jl

release-js:
	./scripts/release_github.sh js

# npm package (delegates to js/Makefile)
js-build:
	$(MAKE) -C js build

js-test:
	$(MAKE) -C js test

jl-install:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'
	julia --project=$(JULIA_PROJECT)/docs -e 'using Pkg; Pkg.develop(path="$(JULIA_PROJECT)"); Pkg.instantiate()'

jl-test:
	julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.test()'

jl-test-slow:
	SCORIO_JL_RUN_SLOW=1 julia --project=$(JULIA_PROJECT) -e 'using Pkg; Pkg.test()'

test-eval-jl:
	julia --project=$(JULIA_PROJECT) -e 'using Scorio; include("$(JULIA_PROJECT)/test/eval/test_eval_apis.jl")'

test-rank-jl:
	julia --project=$(JULIA_PROJECT) -e 'using Scorio; include("$(JULIA_PROJECT)/test/rank/runtests_rank.jl")'

test-rank-jl-slow:
	SCORIO_JL_RUN_SLOW=1 julia --project=$(JULIA_PROJECT) -e 'using Scorio; include("$(JULIA_PROJECT)/test/rank/runtests_rank.jl")'

# Documentation

## Read the Docs
py-docs-build:
	cd docs && make html

py-docs-clean:
	cd docs && make clean

py-docs-serve:
	python -m http.server --directory docs/_build/html 4000

## Julia Docs
jl-docs-build:
	julia --project=$(JULIA_PROJECT)/docs $(JULIA_PROJECT)/docs/make.jl

jl-docs-clean:
	rm -rf $(JULIA_PROJECT)/docs/build

jl-docs-serve:
	python -m http.server --directory $(JULIA_PROJECT)/docs/build 4001

## JavaScript / TypeScript Docs
js-docs-build:
	$(MAKE) -C js docs

js-docs-check:
	$(MAKE) -C js docs-check

js-docs-clean:
	$(MAKE) -C js docs-clean

js-docs-serve:
	cd js/scorio/docs/_build && python3 -m http.server 4003

docs-site-build: jl-docs-build js-docs-build
	rm -rf build/docs-site
	mkdir -p build/docs-site/julia build/docs-site/javascript
	cp -a docs-landing/. build/docs-site/
	cp -a $(JULIA_PROJECT)/docs/build/. build/docs-site/julia/
	cp -a js/scorio/docs/_build/. build/docs-site/javascript/
	touch build/docs-site/.nojekyll
	python scripts/docs_seo.py build/docs-site

docs-seo-check:
	python scripts/docs_seo.py build/docs-site --check

docs-seo-test:
	python -m unittest discover -s tests/docs -v

landing-serve:
	python -m http.server --directory docs-landing 4002
