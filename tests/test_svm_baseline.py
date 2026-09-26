from src.data import load
from src.svm_baseline import run


def test_svm_runs_and_is_deterministic_on_the_sample():
    posts, _ = load("data", sample=True)
    first, second = run(posts), run(posts)
    assert first["micro_f1"] == second["micro_f1"] and first["macro_f1"] == second["macro_f1"]
    assert first["matches_paper"] is False                  # 17 paraphrased posts cannot give 0.39
    assert set(first["split"]) == {"support", "matches_authors"}
