.PHONY: test inspect inspect-sample baseline baseline5 m1 m2 m3 evaluate figures clean

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

# needs data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt, downloaded by hand
m1:
	python -m src.m1

# needs `ollama serve`; resumable, every call is cached
m3:
	python -m src.run_m3

# needs a trained seed-0 checkpoint (make baseline) and `ollama serve` for the decoupled control
m2:
	python -m src.run_m2

evaluate:
	python -m src.evaluate outputs/records/*.jsonl
