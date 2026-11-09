.PHONY: install demo test real
install:
	pip install -r requirements.txt
demo:      ## run the full pipeline on synthetic data (no downloads)
	PYTHONPATH=src python scripts/run_pipeline.py --data synthetic
test:
	PYTHONPATH=src python -m pytest -q tests
real:      ## after running scripts/fetch_data.py
	PYTHONPATH=src python scripts/run_pipeline.py --data data/panel.csv
