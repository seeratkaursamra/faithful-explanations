.PHONY: setup test inspect inspect-sample baseline baseline5 svm m1 m2 m3 m3-lime evaluate evaluate-minilm cases figures clean

# everything installable in one go, inside an active venv; the Ollama pulls need `ollama serve` running.
# The NRC lexicon (licence terms) and ReDSM5 (gated) have to be fetched by hand; see the README.
setup:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"
	ollama pull qwen2.5:7b
	ollama pull nomic-embed-text
	ollama pull all-minilm

test:
	python -m pytest -q tests

inspect:
	python -m src.inspect_data --data data --reading-sample docs/reading_sample.md

# same thing on the public sample (17 posts, 25 annotation rows), where a mismatch with the paper is expected
inspect-sample:
	python -m src.inspect_data --data data --allow-mismatch --reading-sample docs/reading_sample.md

# plausible-but-unfaithful cases for reading by hand; cases.md quotes posts and stays in outputs/
cases:
	python -m src.cases outputs/records/*.jsonl

figures:
	python -m src.figures

clean:
	rm -rf outputs/* docs/reading_sample.md

baseline:
	python -m src.train_bert --config configs/bert_baseline.yaml --seed 0

baseline5:
	python -m src.train_bert --config configs/bert_baseline.yaml --seeds 0 1 2 3 4

# the authors' SVM, a deterministic second reproduction check (their 0.39 / 0.28)
svm:
	python -m src.svm_baseline

# needs data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt, downloaded by hand
m1:
	python -m src.m1

# needs `ollama serve`; resumable, every call is cached
m3:
	python -m src.run_m3

# the overnight job: rewrites m3.jsonl with the same records plus LIME (the joint and erasure calls come
# back from the cache); raise OLLAMA_NUM_PARALLEL and --workers to match
m3-lime:
	python -m src.run_m3 --lime

# needs a trained seed-0 checkpoint (make baseline) and `ollama serve` for the decoupled control;
# --sanity runs the randomised-head check, one of the prereg's conditions for an inconclusive study
m2:
	python -m src.run_m2 --sanity

evaluate:
	python -m src.evaluate outputs/records/*.jsonl

# the prereg's secondary embedding model, everything else the same
evaluate-minilm:
	python -m src.evaluate outputs/records/*.jsonl --embed-model all-minilm --out outputs/metrics_minilm
