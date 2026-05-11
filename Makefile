.PHONY: scrape build-index run test eval probes install

install:
	pip install -r requirements.txt

scrape:
	python catalog/scraper.py

build-index:
	python catalog/build_index.py

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

eval:
	python eval/harness.py

probes:
	python eval/harness.py probes