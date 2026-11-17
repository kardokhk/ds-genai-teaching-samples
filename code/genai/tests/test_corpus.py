"""Corpus and question set: determinism, front matter, no contact details, gold values present in the gold documents."""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


corpus_mod = _load("make_corpus", "01_make_corpus.py")
questions_mod = _load("make_questions", "02_make_questions.py")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    res = corpus_mod.build(out, corpus_mod.SEED)
    rows = questions_mod.build_questions(res["facts"])
    return out, res, rows


def test_deterministic(built, tmp_path):
    out, res, _ = built
    corpus_mod.build(tmp_path, corpus_mod.SEED)
    for m in res["manifest"]:
        assert hashlib.sha256((tmp_path / m["file"]).read_bytes()).hexdigest() == m["sha256"]


def test_documents_and_front_matter(built):
    out, res, _ = built
    docs = sorted((out / "docs").glob("*.md"))
    assert 9 <= len(docs) <= 12
    statuses = []
    for p in docs:
        text = p.read_text()
        head = text.split("---")[1]
        for key in ("doc_id:", "title:", "version:", "status:", "effective_date:"):
            assert key in head, (p.name, key)
        statuses.append(re.search(r"status: (\w+)", head).group(1))
        assert "(fictional)" in text
    assert statuses.count("superseded") == 2
    for m in res["manifest"]:
        assert 380 <= m["words"] <= 1200, m


def test_no_contact_details(built):
    out, _, _ = built
    text = "\n".join(p.read_text() for p in (out / "docs").glob("*.md"))
    assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", text), "email address"
    assert not re.search(r"https?://|www\.", text), "URL"
    assert not re.search(r"(\+\d{2}|\b0\d{3,4})[\s-]?\d{3}[\s-]?\d{3,4}\b", text), "phone number"
    assert not re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}\b", text), "UK postcode"
    assert not re.search(r"\b\d+\s+\w+\s+(Street|Road|Lane|Avenue)\b", text), "street address"


def _flat(items):
    for it in items:
        if isinstance(it, list):
            yield it
        else:
            yield [it]


def test_gold_values_occur_in_gold_documents(built):
    out, _, rows = built
    for r in rows:
        if not r["answerable"] or r["type"] == "injection_bait" and isinstance(r["gold"][0], list) and len(r["gold"][0]) > 3:
            continue
        text = " ".join((out / "docs" / f"{d}.md").read_text() for d in r["gold_doc_ids"]).lower().replace(",", "")
        for alts in _flat(r["gold"]):
            if r["id"] and any(isinstance(a, str) and a in ("not required", "only if") for a in alts):
                continue  # paraphrase alternatives for "no scheduled rotation"
            ok = any((f"{a:.2f}" in text or f"{a:g}" in text) if isinstance(a, (int, float)) else a.lower() in text
                     for a in alts)
            assert ok, (r["id"], alts)


def test_stale_values_are_in_superseded_docs(built):
    out, res, rows = built
    f = res["facts"]
    sup = (out / "docs" / f"{f['leave']['superseded']['doc_id']}.md").read_text() + \
          (out / "docs" / f"{f['it']['superseded']['doc_id']}.md").read_text() + \
          (out / "docs" / f"{f['ops_meeting']['doc_id']}.md").read_text()
    for r in rows:
        for v in r.get("stale", []):
            assert str(v) in sup, (r["id"], v)


def test_split_and_counts(built):
    _, _, rows = built
    assert 55 <= len(rows) <= 70
    dev = [r for r in rows if r["split"] == "dev"]
    assert 0.25 <= len(dev) / len(rows) <= 0.4
    assert 10 <= sum(not r["answerable"] for r in rows) <= 14
    for r in rows:
        assert {"id", "question", "type", "gold", "gold_doc_ids", "answerable", "split"} <= set(r)
        assert bool(r["gold"]) == r["answerable"]
    # no document group appears in both splits
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in groups.values())


def test_canaries_planted(built):
    out, res, _ = built
    for key in ("ops_meeting", "support_meeting"):
        m = res["facts"][key]
        text = (out / "docs" / f"{m['doc_id']}.md").read_text()
        assert m["canaries"][0] in text
