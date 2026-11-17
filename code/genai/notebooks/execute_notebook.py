"""Execute genai_rag_lab.ipynb top to bottom and save the copy with outputs.

Cells tagged "colab-setup" (the clone step) are skipped. Model locations come from the local Hugging Face
cache when HF_HUB_OFFLINE=1, through the same LAB_* variables a learner can set.

    python execute_notebook.py [--out genai_rag_lab.executed.ipynb] [--timing timing.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rag_lab import config as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, default=HERE / "genai_rag_lab.ipynb")
ap.add_argument("--out", type=Path, default=HERE / "genai_rag_lab.executed.ipynb")
ap.add_argument("--timing", type=Path,
                default=HERE.parent.parent.parent / "results" / "2026-11-17_rag_eval" / "notebook_run.json")
ap.add_argument("--cache", type=Path,
                default=HERE.parent.parent.parent / "results" / "2026-11-17_rag_eval" / "cache" / "notebook.jsonl")
ap.add_argument("--scripted", action="store_true",
                help="dry run without a GPU: replace the chat model with a deterministic stand-in")
a = ap.parse_args()

os.environ.setdefault("LAB_CHAT_MODEL", C.chat_model_path("qwen2.5-1.5b"))
os.environ.setdefault("LAB_EMBED_MODEL", C.embed_path())
os.environ.setdefault("LAB_RERANK_MODEL", C.rerank_path())
os.environ["LAB_CACHE"] = str(a.cache)
os.environ["LAB_DIR"] = str(HERE.parent)

nb = nbformat.read(a.src, as_version=4)
if a.scripted:
    stub = ("import os, sys; sys.path.insert(0, os.environ[\"LAB_DIR\"])\n"
            "from rag_lab.chat_model import LocalChatModel\n"
            "from rag_lab.backends import Scripted\n"
            "LocalChatModel.from_model = classmethod(lambda cls, *a, **k: cls(backend=Scripted("
            "lambda s, u: 'NOT_FOUND' if 'Sources' in u else 'I do not know.')))")
    nb.cells.insert(0, nbformat.v4.new_code_cell(stub))
t0 = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="python3", skip_cells_with_tag="colab-setup",
                        resources={"metadata": {"path": str(HERE)}})
err = None
try:
    client.execute()
except Exception as e:  # keep the partial notebook for inspection
    err = f"{type(e).__name__}: {str(e)[-800:]}"
elapsed = round(time.time() - t0, 1)
if a.scripted:
    nb.cells.pop(0)
# keep learner-facing output only: drop warnings on stderr, then refuse to save if a local path survives
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs = [o for o in c.get("outputs", []) if not (o.get("output_type") == "stream" and o.get("name") == "stderr")]
text = nbformat.writes(nb)
private = {str(Path.home()), str(HERE.parent.parent.parent), sys.prefix, os.environ.get("HF_HOME", "") or "\0",
           os.environ.get("USER", "") or "\0"}
leaks = sorted(x for x in private if x and x != "\0" and x in text)
if leaks:
    err = (err or "") + f" local paths or names in outputs: {len(leaks)}"
    a.out = a.out.with_suffix(".unsafe.ipynb")
nbformat.write(nb, a.out)
a.timing.parent.mkdir(parents=True, exist_ok=True)
a.timing.write_text(json.dumps({"elapsed_seconds": elapsed, "error": err, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                "cells": len(nb.cells)}, indent=1) + "\n")
print("executed in", elapsed, "s; error:", err)
sys.exit(1 if err else 0)
