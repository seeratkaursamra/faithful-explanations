.PHONY: test inspect inspect-sample clean

test:
	python -m pytest -q tests

inspect:
	python -m src.inspect_data --data data --reading-sample docs/reading_sample.md

# same thing on the public 25-row sample, where a mismatch with the paper is expected
inspect-sample:
	python -m src.inspect_data --data data --allow-mismatch --reading-sample docs/reading_sample.md

clean:
	rm -rf outputs/* docs/reading_sample.md
