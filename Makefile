.PHONY: scrape build-index run test eval probes install

install:
	pip install -r requirements.txt

scrape:
	python -m catalog.scraper

build-index:
	python -m catalog.build_index

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

eval:
	python -m eval.harness

probes:
	python -m eval.harness probes