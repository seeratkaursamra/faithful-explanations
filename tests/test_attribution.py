from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from src.attribution import baseline_ids, explain, integrated_gradients
from src.sentences import split

WORDS = ["i", "cannot", "sleep", "my", "appetite", "is", "gone", ".", "tired"]


def tokenizer(tmp_path):
    """A real BERT tokenizer over a ten-word vocabulary, so nothing is downloaded."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS))
    return BertTokenizer(vocab_file=str(vocab))


def tiny_bert(n_labels=3):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=64, num_labels=n_labels)
    return BertForSequenceClassification(config)


class Linear(torch.nn.Module):
    """logit = w . (sum of the token embeddings). IG on a linear model is exact: token i gets (e_i - e_pad) . w."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(6, 4)
        self.w = torch.nn.Parameter(torch.randn(4, 2))

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask):
        return SimpleNamespace(logits=inputs_embeds.sum(dim=1) @ self.w)


def test_ig_is_exact_on_a_linear_model():
    model = Linear()
    ids, pad = torch.tensor([[1, 4, 5, 2]]), torch.tensor([[0, 0, 0, 0]])
    attributions, delta = integrated_gradients(model, ids, torch.ones_like(ids), pad, target=1, steps=5)
    e = model.embed.weight.detach()
    expected = [float((e[i] - e[0]) @ model.w[:, 1].detach()) for i in [1, 4, 5, 2]]
    assert attributions == pytest.approx(expected, abs=1e-5)
    assert delta == pytest.approx(0.0, abs=1e-5)


def test_ig_on_bert_sums_to_the_change_in_the_logit(tmp_path):
    tok, model = tokenizer(tmp_path), tiny_bert()
    enc = tok("i cannot sleep . my appetite is gone .", return_tensors="pt")
    base = baseline_ids(enc["input_ids"], tok)
    _, rough = integrated_gradients(model, enc["input_ids"], enc["attention_mask"], base, target=2, steps=4)
    _, fine = integrated_gradients(model, enc["input_ids"], enc["attention_mask"], base, target=2, steps=200)
    assert abs(fine) < 1e-3
    assert abs(fine) < abs(rough)


def test_baseline_keeps_cls_and_sep_and_blanks_everything_else(tmp_path):
    tok = tokenizer(tmp_path)
    ids = tok("i cannot zzz", return_tensors="pt")["input_ids"]     # zzz is [UNK], a real word
    assert tok.convert_ids_to_tokens(baseline_ids(ids, tok)[0]) == ["[CLS]", "[PAD]", "[PAD]", "[PAD]", "[SEP]"]


def test_cls_and_sep_get_zero_attribution(tmp_path):
    tok, model = tokenizer(tmp_path), tiny_bert()
    enc = tok("i cannot sleep .", return_tensors="pt")
    attributions, _ = integrated_gradients(model, enc["input_ids"], enc["attention_mask"],
                                           baseline_ids(enc["input_ids"], tok), target=0, steps=10)
    assert attributions[0] == 0.0 and attributions[-1] == 0.0
    assert any(a != 0.0 for a in attributions[1:-1])


def test_explain_scores_every_sentence_and_nothing_else(tmp_path):
    tok, model = tokenizer(tmp_path), tiny_bert()
    text = "I cannot sleep. My appetite is gone. Tired."
    scores, delta = explain(model, tok, text, split(text), target=1, steps=20)
    assert set(scores) == {0, 1, 2}


def test_explain_sentence_scores_keep_igs_completeness(tmp_path):
    # the specials get 0, so the sentences must add up to exactly what the tokens add up to
    tok, model = tokenizer(tmp_path), tiny_bert()
    text = "I cannot sleep. My appetite is gone. Tired."
    scores, _ = explain(model, tok, text, split(text), target=1, steps=20)
    enc = tok(text, return_tensors="pt")
    tokens, _ = integrated_gradients(model, enc["input_ids"], enc["attention_mask"],
                                     baseline_ids(enc["input_ids"], tok), target=1, steps=20)
    assert sum(scores.values()) == pytest.approx(sum(tokens), abs=1e-6)


def test_randomised_head_redraws_only_the_head_and_leaves_the_original_alone():
    from src.attribution import randomised_head
    model = tiny_bert()
    before = {n: p.clone() for n, p in model.named_parameters()}
    random = randomised_head(model, seed=1)
    for name, p in random.named_parameters():
        same = torch.equal(p, before[name])
        assert same != name.startswith("classifier.weight"), name     # only the head's weights changed
    assert torch.equal(random.classifier.bias, torch.zeros_like(random.classifier.bias))
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())   # the original is untouched
    assert torch.equal(randomised_head(model, 1).classifier.weight, random.classifier.weight)   # seeded
    assert abs(float(random.classifier.weight.detach().std()) - 0.02) < 0.01      # drawn at BERT's own scale
