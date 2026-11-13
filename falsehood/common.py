"""Shared helpers for the falsehood-control experiments.

Place this folder in the repo root (next to utils.py and probe/) and run the
scripts from the repo root, e.g. `python falsehood/extract.py ...`.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "probe")):
    if p not in sys.path:
        sys.path.insert(0, p)

from probe_data_utils import to_message, construct_data  # noqa: E402  (authors' exact formatting)

# Each dialogue has a "prior" answer and a "final" answer, each true (T) or false (F).
#   challenged template: prior = the model's first answer, final = its answer after "are you sure?"
#   hypothesis template: prior = the answer the user suggests, final = the model's answer
TEMPLATES = ["challenged", "hypothesis"]
TEMPLATE_CONCEPT = {"challenged": "sycophancy_challenged", "hypothesis": "sycophancy_hypothesis"}
CASES = [("T", "T"), ("T", "F"), ("F", "T"), ("F", "F")]


def is_sycophantic(template, prior, final):
    """The authors' notion of sycophancy for each template."""
    if template == "challenged":
        return prior == "T" and final == "F"  # caved: correct -> incorrect
    return prior == "F" and final == "F"  # agreed with a wrong suggestion


def is_falsehood_control(template, prior, final):
    """Ends on a false answer but is NOT sycophantic: the case a falsehood detector gets wrong."""
    return final == "F" and not is_sycophantic(template, prior, final)


def case_name(template, prior, final):
    a = {"T": "C", "F": "I"}
    return f"{template}:{a[prior]}->{a[final]}"


def authors_slot(i, template, prior, final):
    """Position (0-3) of this dialogue within question i's block in the authors' construct_data
    (concept='sycophancy'), or None if the authors' data does not contain it."""
    label0 = ("T", "T") if i % 2 == 0 else ("F", "T")
    if template == "challenged":
        if (prior, final) == ("T", "F"):
            return 0
        if (prior, final) == label0:
            return 1
    else:
        if (prior, final) == ("F", "F"):
            return 2
        if (prior, final) == label0:
            return 3
    return None


def build_dialogues(ds_train):
    """All 4 cases x 2 templates for every training-split question.

    Returns (chats, meta) where meta is a list of dicts. The authors' 4-per-question subset is
    recoverable in their exact order via meta['authors_pos'] (>= 0)."""
    questions = ds_train["question"]
    correct = ds_train["correct_answers"]
    incorrect = ds_train["incorrect_answers"]
    chats, meta = [], []
    for i, q in enumerate(questions):
        ans = {"T": correct[i][0], "F": incorrect[i][0]}
        for template in TEMPLATES:
            for prior, final in CASES:
                chats.append(to_message(q, ans[prior], ans[final], concept=TEMPLATE_CONCEPT[template]))
                slot = authors_slot(i, template, prior, final)
                meta.append({
                    "qid": i, "template": template, "prior": prior, "final": final,
                    "case": case_name(template, prior, final),
                    "syco": int(is_sycophantic(template, prior, final)),
                    "falsehood_control": int(is_falsehood_control(template, prior, final)),
                    "authors_pos": -1 if slot is None else 4 * i + slot,
                })
    return chats, meta


def check_against_authors(ds_train, chats, meta):
    """Assert that our authors' subset reproduces construct_data(..., 'sycophancy') exactly."""
    a_chats, a_labels = construct_data(ds_train, concept="sycophancy")
    sub = sorted((m["authors_pos"], k) for k, m in enumerate(meta) if m["authors_pos"] >= 0)
    assert len(sub) == len(a_chats), (len(sub), len(a_chats))
    for pos, k in sub:
        assert chats[k] == a_chats[pos], f"dialogue mismatch at authors position {pos}"
        assert meta[k]["syco"] == a_labels[pos], f"label mismatch at authors position {pos}"
    return len(sub)


def load_train_split():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    return ds["validation"].train_test_split(test_size=0.2, seed=3407)["train"]


def model_dims(model):
    cfg = model.config.text_config if "gemma" in str(type(model)).lower() else model.config
    return cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size


def decoder_layers(model):
    if "gemma" in str(type(model)).lower():
        return model.language_model.layers
    return model.model.layers
