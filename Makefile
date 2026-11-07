.PHONY: twin test run serve
twin:; PYTHONPATH=src python -m stylematch.twin
run: twin
	PYTHONPATH=src python scripts/run_match.py && PYTHONPATH=src python scripts/run_attributes.py && PYTHONPATH=src python scripts/run_price.py
test:; PYTHONPATH=src python -m pytest -q
serve:; PYTHONPATH=src uvicorn stylematch.api:app --port 8000
real:; PYTHONPATH=src python scripts/run_shopee.py && PYTHONPATH=src python scripts/run_hm_attributes.py && PYTHONPATH=src python scripts/run_hm_prices.py
