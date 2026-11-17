"""Check the grader against hand scores.

sample   draw 10 outputs per pipeline (seeded) from a run directory and print them for hand scoring
compare  compare hand labels with the grader and write grader_check.json

Hand labels file (JSON): {"<model>|<pipeline>|<id>": "good" | "bad", ...}, where "good" means a correct
answer for an answerable question or a refusal for an unanswerable one (the grader's good_outcome).

    python 05_grader_check.py sample --run-dir RUN
    python 05_grader_check.py compare --run-dir RUN --labels RUN/hand_labels.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(run: Path):
    recs = []
    for f in sorted(run.glob("outputs_*_*.jsonl")):
        recs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return recs


def good(r):
    g = r["grade"]
    return bool(g["correct"]) if r["answerable"] else bool(g["correct_refusal"])


def sample(recs, per=10, seed=20261117):
    rng = random.Random(seed)
    out = []
    for p in ("closed_book", "naive_rag", "improved_rag"):
        pool = sorted([r for r in recs if r["pipeline"] == p], key=lambda r: (r["model"], r["id"]))
        out += rng.sample(pool, min(per, len(pool)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "compare"])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--regrade", action="store_true", help="grade again with the current grader")
    ap.add_argument("--seed", type=int, default=20261117, help="sampling seed (20261118 for the test-split check)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--corpus", type=Path, default=HERE.parent.parent / "data" / "processed" / "genai_corpus")
    a = ap.parse_args()
    picked = sample(load(a.run_dir), seed=a.seed)
    if a.regrade:
        from rag_lab.corpus import load_questions
        from rag_lab.grader import grade
        qs = {q["id"]: q for q in load_questions(a.corpus)}
        for r in picked:
            r["grade"] = grade(qs[r["id"]], r["answer"], r["sources"])
    if a.cmd == "sample":
        for r in picked:
            print(f"== {r['model']}|{r['pipeline']}|{r['id']} [{r['type']}] grader={'good' if good(r) else 'bad'}")
            print("Q:", r["question"])
            print("A:", r["answer"].replace("\n", " ")[:600])
        return
    labels = json.loads(a.labels.read_text())
    rows, dis = {}, []
    for r in picked:
        k = f"{r['model']}|{r['pipeline']}|{r['id']}"
        hand = labels[k] == "good"
        st = rows.setdefault(r["pipeline"], {"n": 0, "agree": 0})
        st["n"] += 1
        st["agree"] += hand == good(r)
        if hand != good(r):
            dis.append({"key": k, "hand": labels[k], "grader": "good" if good(r) else "bad", "answer": r["answer"]})
    res = {"run": a.run_dir.name, "per_pipeline": rows, "disagreements": dis,
           "total_agree": sum(v["agree"] for v in rows.values()), "total_n": sum(v["n"] for v in rows.values())}
    res["grader"] = "current (regraded)" if a.regrade else "as recorded in the run"
    out = a.out or a.run_dir / "grader_check.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}), len(dis), "disagreements ->", out)


if __name__ == "__main__":
    main()
