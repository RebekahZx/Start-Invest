.PHONY: install train serve demo test e2e lint format docker docker-run clean

install:        ## install the package with dev tools
	pip install -e ".[dev]"

train:          ## train all models, write artifacts/ and print the results tables
	start-invest-train --out artifacts

serve:          ## run the API + UI at http://localhost:8000
	uvicorn start_invest.api:app --reload --port 8000

demo:           ## refresh the static snapshot used by GitHub Pages
	start-invest-export-demo --bundle artifacts/bundle.joblib --out web/demo-data.json

test:           ## unit + API tests
	pytest -q

e2e:            ## browser tests (needs: python -m playwright install chromium)
	RUN_E2E=1 pytest -q -m e2e

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff check --fix src tests && ruff format src tests

docker:
	docker build -t start-invest .

docker-run:
	docker run --rm -p 8000:8000 start-invest

clean:
	rm -rf artifacts/*.joblib .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
