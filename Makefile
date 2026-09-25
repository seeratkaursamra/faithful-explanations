.PHONY: test inspect inspect-sample baseline baseline5 figures clean

test:
	python -m pytest -q tests

inspect:
	python -m src.inspect_data --data data --reading-sample docs/reading_sample.md

# same thing on the public 25-row sample, where a mismatch with the paper is expected
inspect-sample:
	python -m src.inspect_data --data data --allow-mismatch --reading-sample docs/reading_sample.md

figures:
	python -m src.figures

clean:
	rm -rf outputs/* docs/reading_sample.md

baseline:
	python -m src.train_bert --config configs/bert_baseline.yaml --seed 0

baseline5:
	python -m src.train_bert --config configs/bert_baseline.yaml --seeds 0 1 2 3 4
