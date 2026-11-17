"""Run and grade the three RAG pipelines, then summarise with intervals and a paired test.

Subcommands
    run        answer one split with one model and all pipelines; writes per-question outputs
    summarise  grade-level summary of a run directory: rates with Wilson 95% intervals per model and
               pipeline, and naive_rag versus improved_rag on the same questions (exact McNemar P and
               the paired difference with a Newcombe 95% interval)

Backends
    hf         load the model with transformers (GPU), cache every reply
    replay     replay every reply from the cache on a CPU; any missing reply is an error
    scripted   deterministic stand-in replies, for dry runs and tests

Examples
    python 03_evaluate.py run --split dev --model qwen2.5-1.5b --backend scripted --limit 3 --run-dir RUN
    python 03_evaluate.py run --split test --model qwen2.5-7b --backend hf --run-dir RUN
    python 03_evaluate.py run --split test --model qwen2.5-7b --backend replay --run-dir RUN2
    python 03_evaluate.py summarise --run-dir RUN --out RUN/summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rag_lab import config as C  # noqa: E402
from rag_lab.grader import grade  # noqa: E402
from rag_lab.stats import mcnemar_exact, newcombe_paired_diff, wilson  # noqa: E402

ROOT = HERE.parent.parent
DEFAULT_CORPUS = ROOT / "data" / "processed" / "genai_corpus"
DEFAULT_CACHE = ROOT / "results" / "2026-11-17_rag_eval" / "cache"

METRICS = [  # (name, flag, which questions form the denominator)
    ("correct_answerable", "correct", "answerable"),
    ("correct_refusal_unanswerable", "correct_refusal", "unanswerable"),
    ("good_outcome_all", "good", "all"),
    ("confident_wrong_all", "confident_wrong", "all"),
    ("cited_gold_answerable", "cited_gold", "answerable"),
    ("retrieval_hit_all", "retrieval_hit", "all"),
    ("superseded_error", "superseded_error", "stale"),
    ("injection_followed", "injection_followed", "bait"),
]


def scripted_reply(system: str, user: str) -> str:
    """Stand-in model: quotes the first number of the first source, or says it does not know."""
    if "Sources:" in user or "Context:" in user:
        body = user.split("Question:")[0]
        m = re.search(r"\b\d+(?:\.\d+)?\b", body.split("\n", 2)[-1])
        return f"The value is {m.group(0)} [S1]." if m else "NOT_FOUND"
    return "I do not know Larkspur's internal policies."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_llm(model: str, backend: str, cache_dir: Path):
    from rag_lab.backends import Cached, Scripted, model_key
    from rag_lab.chat_model import LocalChatModel
    cache = cache_dir / f"{model}.jsonl"
    if backend == "scripted":
        return LocalChatModel(backend=Cached(Scripted(scripted_reply), cache_dir / "scripted.jsonl"),
                              max_tokens=C.MAX_NEW_TOKENS)
    path = C.chat_model_path(model)
    if backend == "replay":
        return LocalChatModel(backend=Cached(None, cache, offline=True, name=model_key(path)),
                              max_tokens=C.MAX_NEW_TOKENS)
    return LocalChatModel.from_model(path, cache_path=cache, max_tokens=C.MAX_NEW_TOKENS, seed=C.SEED)


def cmd_run(a):
    import random

    import numpy as np
    import torch

    from rag_lab.corpus import load_documents, load_questions
    from rag_lab.pipelines import build_all

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    t0 = time.time()
    rows = load_questions(a.corpus, a.split)
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        raise SystemExit("no questions selected")
    docs = load_documents(a.corpus)
    llm = make_llm(a.model, a.backend, a.cache_dir)
    t_load = time.time() - t0
    pipes = build_all(llm, docs, names=a.pipelines)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name, pipe in pipes.items():
        t1 = time.time()
        res = pipe.answer([r["question"] for r in rows])
        timings[name] = round(time.time() - t1, 1)
        with (a.run_dir / f"outputs_{a.model}_{name}.jsonl").open("w") as fh:
            for r, out in zip(rows, res):
                rec = {"id": r["id"], "model": a.model, "pipeline": name, "split": r["split"], "type": r["type"],
                       "answerable": r["answerable"], "question": r["question"], "answer": out["answer"],
                       "sources": out["sources"], "grade": grade(r, out["answer"], out["sources"])}
                fh.write(json.dumps(rec) + "\n")
        print(f"{a.model} {name}: {len(rows)} questions in {timings[name]} s", flush=True)
    be = llm.backend
    meta = {"model": a.model, "backend": a.backend, "split": a.split, "n_questions": len(rows),
            "model_repo": C.MODELS.get(a.model, ("scripted", ""))[0], "model_revision": C.MODELS.get(a.model, ("", ""))[1],
            "embed": list(C.EMBED), "rerank": list(C.RERANK), "seed": C.SEED, "max_new_tokens": C.MAX_NEW_TOKENS,
            "cache_hits": be.hits, "cache_misses": be.misses, "seconds_load": round(t_load, 1),
            "seconds_per_pipeline": timings, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu": (__import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None),
            "python": platform.python_version(),
            "questions_sha256": sha(Path(a.corpus) / "questions.jsonl"),
            "corpus_manifest_sha256": sha(Path(a.corpus) / "manifest.json")}
    (a.run_dir / f"meta_{a.model}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"cache hits {be.hits}, misses {be.misses}; total {round(time.time() - t0, 1)} s")


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def in_denominator(rec: dict, which: str) -> bool:
    g = rec["grade"]
    return {"all": True, "answerable": rec["answerable"], "unanswerable": not rec["answerable"],
            "stale": g["superseded_error"] is not None, "bait": g["injection_followed"] is not None}[which]


def flag(rec: dict, name: str) -> bool:
    g = rec["grade"]
    if name == "good":
        return bool(g["correct"]) if rec["answerable"] else bool(g["correct_refusal"])
    return bool(g[name])


def cmd_summarise(a):
    run = a.run_dir
    files = sorted(run.glob("outputs_*_*.jsonl"))
    if not files:
        raise SystemExit(f"no outputs in {run}")
    by = {}
    for f in files:
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        by[(recs[0]["model"], recs[0]["pipeline"])] = recs
    models = sorted({m for m, _ in by}, key=lambda m: list(C.MODELS).index(m) if m in C.MODELS else 99)
    out = {"run_id": a.run_id or run.name, "split": None, "seed": C.SEED, "metrics": {}, "paired": {},
           "models": {}, "definitions": {m: f"{fl} over {den} questions" for m, fl, den in METRICS}}
    for m in models:
        meta_p = run / f"meta_{m}.json"
        out["models"][m] = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        out["metrics"][m] = {}
        for p in ("closed_book", "naive_rag", "improved_rag"):
            recs = by.get((m, p))
            if not recs:
                continue
            out["split"] = recs[0]["split"]
            out["metrics"][m][p] = {}
            for name, fl, den in METRICS:
                sel = [r for r in recs if in_denominator(r, den)]
                if fl == "retrieval_hit" and p == "closed_book":
                    continue
                out["metrics"][m][p][name] = rate(sum(flag(r, fl) for r in sel), len(sel))
        # paired: naive_rag (A) versus improved_rag (B) on identical questions
        A, B = by.get((m, "naive_rag")), by.get((m, "improved_rag"))
        if A and B:
            ia, ib = {r["id"]: r for r in A}, {r["id"]: r for r in B}
            ids = sorted(set(ia) & set(ib))
            assert ids, "paired comparison is empty"
            out["paired"][m] = {}
            for name, fl, den in [("good_outcome_all", "good", "all"), ("correct_answerable", "correct", "answerable"),
                                  ("confident_wrong_all", "confident_wrong", "all")]:
                sel = [i for i in ids if in_denominator(ia[i], den)]
                pa = [flag(ia[i], fl) for i in sel]
                pb = [flag(ib[i], fl) for i in sel]
                both = sum(x and y for x, y in zip(pa, pb))
                a_only = sum(x and not y for x, y in zip(pa, pb))
                b_only = sum(y and not x for x, y in zip(pa, pb))
                neither = len(sel) - both - a_only - b_only
                d, lo, hi = newcombe_paired_diff(both, a_only, b_only, neither)
                out["paired"][m][name] = {"n": len(sel), "naive_rag": sum(pa), "improved_rag": sum(pb),
                                          "naive_only": a_only, "improved_only": b_only,
                                          "mcnemar_exact_p": round(mcnemar_exact(a_only, b_only), 4),
                                          "diff_improved_minus_naive": round(d, 4),
                                          "diff_ci95_newcombe": [round(lo, 4), round(hi, 4)]}
    out["slurm_job_ids"] = sorted({v.get("slurm_job_id") for v in out["models"].values() if v.get("slurm_job_id")})
    if a.extra:
        out.update(json.loads(Path(a.extra).read_text()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for m, ps in out["metrics"].items():
        for p, ms in ps.items():
            g, w = ms["good_outcome_all"], ms["confident_wrong_all"]
            print(f"{m:13s} {p:13s} good {g['k']}/{g['n']}  confident wrong {w['k']}/{w['n']}")
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test", "all"], required=True)
    r.add_argument("--model", choices=list(C.MODELS) + ["scripted"], required=True)
    r.add_argument("--backend", choices=["hf", "replay", "scripted"], required=True)
    r.add_argument("--pipelines", nargs="+", default=["closed_book", "naive_rag", "improved_rag"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    r.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    r.add_argument("--run-dir", type=Path, required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--run-id")
    s.add_argument("--extra", help="JSON file merged into the summary (job ids, elapsed times)")
    a = ap.parse_args()
    {"run": cmd_run, "summarise": cmd_summarise}[a.cmd](a)


if __name__ == "__main__":
    main()
