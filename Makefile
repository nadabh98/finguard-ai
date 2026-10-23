install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
train:
	python -m finguard.train
test:
	pytest -q
