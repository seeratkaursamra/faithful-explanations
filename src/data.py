"""Load ReDSM5 the way the authors' notebook does. The same call and seed give the same split only if the
rows come in the same order as their redsm5.csv, so check_split compares the result with their test set."""
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

# The Hugging Face release we use (irlab-udc/redsm5, commit b81b534, updated 2026-07-06, after the
# paper) has one more suicidal-thoughts post than the paper and its own dataset card; the rows are clean
# (checked). This is the only difference the gate accepts; anything else still stops training.
RELEASE = "irlab-udc/redsm5@b81b5348993eacfca8eb0d67d6c2be04d0ff3931"
KNOWN_DIFFERENCES = {"SUICIDAL_THOUGHTS": 1}

# Per-label support in the authors' test split, printed by their examples/svm.ipynb. Matching all ten
# is strong evidence our split holds the same 297 posts, which the call and seed alone cannot promise.
AUTHORS_TEST_SUPPORT = {
    "ANHEDONIA": 25, "APPETITE_CHANGE": 10, "COGNITIVE_ISSUES": 10, "DEPRESSED_MOOD": 70, "FATIGUE": 28,
    "NO_SYMPTOMS": 73, "PSYCHOMOTOR": 8, "SLEEP_ISSUES": 18, "SUICIDAL_THOUGHTS": 28, "WORTHLESSNESS": 72,
}


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


def gold_text(ann, post_id, symptom):
    """G_text: the psychologist's explanations for one (post, symptom), joined in sentence order."""
    rows = gold(ann, post_id, symptom).sort_values("sentence_index")
    return " ".join(e.strip() for e in rows["explanation"] if isinstance(e, str))


def rejection_text(ann, post_id, symptom):
    """The annotator's notes on sentences considered and rejected for this symptom in this post (status 0),
    joined in sentence order; empty when there are none."""
    rows = ann[(ann["post_id"] == post_id) & (ann["DSM5_symptom"] == symptom) & (ann["status"] == 0)]
    return " ".join(e.strip() for e in rows.sort_values("sentence_index")["explanation"] if isinstance(e, str))


def references(ann, post_ids):
    """Every G_text per symptom over the given posts. Pass training post ids only, so no test post
    ever ends up in the set it is compared against."""
    pos = ann[(ann["status"] == 1) & ann["post_id"].isin(set(post_ids)) & ann["DSM5_symptom"].isin(SYMPTOMS)]
    out = {}
    for (post_id, symptom), _ in pos.groupby(["post_id", "DSM5_symptom"], sort=True):
        text = gold_text(ann, post_id, symptom)
        if text:
            out.setdefault(symptom, []).append(text)
    return out


def symptom_counts(posts):
    counts = {}
    for labs in posts["labels"]:
        for lab in labs:
            if lab != NO_SYMPTOMS:
                counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def check_split(test):
    """Our test split's per-label support against the authors'. {label: (ours, theirs)} and whether all match."""
    ours = {label: 0 for label in AUTHORS_TEST_SUPPORT}
    for labs in test["labels"]:
        for lab in labs:
            ours[lab] = ours.get(lab, 0) + 1
    support = {label: (ours.get(label, 0), n) for label, n in AUTHORS_TEST_SUPPORT.items()}
    return {"support": support, "matches_authors": all(a == b for a, b in support.values())}


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
        and all(ours - paper in (0, KNOWN_DIFFERENCES.get(s, 0)) for s, (ours, paper) in report["symptom_counts"].items())
    )
    report["known_differences"] = {"release": RELEASE, "accepted": KNOWN_DIFFERENCES}
    return report
