.PHONY: install train serve dev lint test clean

install:
	uv sync
	cd ui && npm install

train:
	uv run sentirise train --epochs 1 --max-train 500 --max-eval 100

serve:
	uv run sentirise serve

dev:
	cd ui && npm run dev

lint:
	uv run ruff check src tests
	cd ui && npm run lint

test:
	uv run pytest -v

clean:
	rm -rf .pytest_cache .ruff_cache data/adapters data/model_output
	rm -rf ui/.next
