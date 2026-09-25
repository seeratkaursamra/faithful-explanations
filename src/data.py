"""Load ReDSM5 the way the authors' notebook does, so the split is identical everywhere."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

SYMPTOMS = [
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
]
NO_SYMPTOMS = "NO_SYMPTOMS"

# posts per symptom (status == 1) reported in the paper
PAPER_COUNTS = {
    "DEPRESSED_MOOD": 328,
    "WORTHLESSNESS": 311,
    "SUICIDAL_THOUGHTS": 165,
    "FATIGUE": 124,
    "ANHEDONIA": 124,
    "SLEEP_ISSUES": 102,
    "COGNITIVE_ISSUES": 59,
    "APPETITE_CHANGE": 44,
    "PSYCHOMOTOR": 35,
}
PAPER_N_POSTS = 1484
PAPER_N_NEGATIVES = 392


def load(data_dir="data", sample=False):
    """Return (posts, annotations); posts gets a labels column with the notebook's 10-class scheme.

    sample=True reads only the public sample, even when the real files are present. Tests use it.
    """
    data_dir = Path(data_dir)
    posts_names = ["redsm5_posts_sample_paraphrased.csv"]
    ann_names = ["redsm5_annotations_sample_paraphrased.csv"]
    if not sample:
        posts_names.insert(0, "redsm5_posts.csv")
        ann_names.insert(0, "redsm5_annotations.csv")
    posts = pd.read_csv(_find(data_dir, *posts_names), dtype={"post_id": str, "text": str})
    ann = pd.read_csv(_find(data_dir, *ann_names),
                      dtype={"post_id": str, "sentence_id": str, "DSM5_symptom": str})
    ann["status"] = ann["status"].astype(int)
    ann["sentence_index"] = ann["sentence_id"].str.rsplit("_", n=1).str[-1].astype(int)

    # a post's labels are its status==1 symptoms; a post with none is NO_SYMPTOMS
    pos = ann[(ann["status"] == 1) & ann["DSM5_symptom"].isin(SYMPTOMS)]
    post_to_syms = pos.groupby("post_id")["DSM5_symptom"].agg(lambda s: sorted(set(s)))
    posts["labels"] = [post_to_syms.get(pid, [NO_SYMPTOMS]) for pid in posts["post_id"]]
    posts["n_symptoms"] = [0 if labs == [NO_SYMPTOMS] else len(labs) for labs in posts["labels"]]
    posts["is_negative"] = posts["n_symptoms"] == 0
    posts["n_words"] = posts["text"].fillna("").str.split().str.len()
    return posts, ann


def _find(data_dir, *names):
    for name in names:
        if (data_dir / name).exists():
            return data_dir / name
    raise FileNotFoundError(f"none of {names} in {data_dir}; see README for getting the data")


def split(posts):
    """Same call and seed as the authors' notebook. Not stratified."""
    train, test = train_test_split(posts, test_size=0.2, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def binarizer(posts):
    """Fit on all posts, like the notebook. Classes come out sorted alphabetically."""
    return MultiLabelBinarizer().fit(posts["labels"])


def gold(ann, post_id, symptom):
    """The psychologist's evidence sentences and rationales for one (post, symptom)."""
    return ann[(ann["post_id"] == post_id) & (ann["DSM5_symptom"] == symptom) & (ann["status"] == 1)]


def symptom_counts(posts):
    counts = {}
    for labs in posts["labels"]:
        for lab in labs:
            if lab != NO_SYMPTOMS:
                counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def check_against_paper(posts):
    counts = symptom_counts(posts)
    report = {
        "n_posts": (len(posts), PAPER_N_POSTS),
        "n_negatives": (int(posts["is_negative"].sum()), PAPER_N_NEGATIVES),
        "symptom_counts": {s: (counts.get(s, 0), PAPER_COUNTS[s]) for s in SYMPTOMS},
    }
    report["matches_paper"] = (
        report["n_posts"][0] == PAPER_N_POSTS
        and report["n_negatives"][0] == PAPER_N_NEGATIVES
        and all(ours == paper for ours, paper in report["symptom_counts"].values())
    )
    return report
