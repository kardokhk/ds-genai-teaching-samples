"""Figure of the frozen test run: rate with Wilson 95% interval per model and pipeline.

Reads results/2026-11-17_rag_eval/summary.json and writes figures/genai/rag_eval.png (300 dpi),
rag_eval.pdf and rag_eval.csv (the plotted values) at 160 mm wide.

    python 04_figure.py [--summary PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
os.environ.setdefault("SOURCE_DATE_EPOCH", "1794873600")  # 2026-11-17, byte-stable PDF

OKABE_ITO = {"qwen2.5-1.5b": "#0072B2", "qwen2.5-7b": "#D55E00"}
LABEL = {"qwen2.5-1.5b": "Qwen2.5-1.5B-Instruct", "qwen2.5-7b": "Qwen2.5-7B-Instruct"}
PIPES = [("closed_book", "Closed book"), ("naive_rag", "Naive RAG"), ("improved_rag", "Improved RAG")]
PANELS = [("correct_answerable", "Correct answer\n(answerable questions)"),
          ("correct_refusal_unanswerable", "Correct refusal\n(unanswerable questions)"),
          ("confident_wrong_all", "Confident wrong answer\n(all questions; lower is better)"),
          ("cited_gold_answerable", "Cites the right document\n(answerable questions)")]
MM = 1 / 25.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=ROOT / "results" / "2026-11-17_rag_eval" / "summary.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "genai")
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    models = list(s["metrics"])
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Arimo", "Liberation Sans", "Arial",
                            "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, len(PANELS), figsize=(160 * MM, 62 * MM), sharey=True)
    rows = []
    offs = {m: ((len(models) - 1) / 2 - i) * 0.22 for i, m in enumerate(models)}
    for ax, (metric, title) in zip(axes, PANELS):
        ns = set()
        for m in models:
            for y, (p, _) in enumerate(PIPES):
                v = s["metrics"][m].get(p, {}).get(metric)
                if not v:
                    continue
                ns.add(v["n"])
                yy = len(PIPES) - 1 - y + offs[m]
                lo, hi = v["ci95"]
                ax.plot([100 * lo, 100 * hi], [yy, yy], color=OKABE_ITO[m], lw=1.0, solid_capstyle="butt")
                ax.plot(100 * v["rate"], yy, "o", ms=3.5, color=OKABE_ITO[m], mec="white", mew=0.4)
                rows.append({"metric": metric, "model": m, "pipeline": p, "k": v["k"], "n": v["n"],
                             "rate": v["rate"], "ci95_low": lo, "ci95_high": hi})
        ax.set_title(title + f"\nn = {'/'.join(map(str, sorted(ns)))}", pad=4)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("% of questions")
        ax.grid(axis="x", color="#d9d9d9", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(PIPES)))
    axes[0].set_yticklabels([lab for _, lab in reversed(PIPES)])
    axes[0].set_ylim(-0.6, len(PIPES) - 0.4)
    handles = [plt.Line2D([], [], color=OKABE_ITO[m], marker="o", ms=3.5, lw=1.0, label=LABEL[m]) for m in models]
    fig.legend(handles=handles, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.55, 1.0),
               title="Dot: rate on the held-out test questions; line: 95% interval (Wilson)", title_fontsize=7)
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.17, top=0.70, wspace=0.25)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_dir / "rag_eval.png", dpi=300)
    fig.savefig(a.out_dir / "rag_eval.pdf", metadata={"CreationDate": None, "ModDate": None})
    with (a.out_dir / "rag_eval.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # markdown table of the same numbers, for the README
    def cell(v):
        return f"{v['k']}/{v['n']} ({100 * v['rate']:.0f}%, {100 * v['ci95'][0]:.0f} to {100 * v['ci95'][1]:.0f})"
    cols = [("correct_answerable", "Correct (answerable)"), ("correct_refusal_unanswerable", "Refuses (unanswerable)"),
            ("confident_wrong_all", "Confident wrong (all)"), ("cited_gold_answerable", "Cites right document"),
            ("superseded_error", "Old policy value"), ("injection_followed", "Follows injection")]
    lines = ["| Model | Pipeline | " + " | ".join(c for _, c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for m in models:
        for p, lab in PIPES:
            ms = s["metrics"][m].get(p)
            if ms:
                lines.append(f"| {LABEL[m]} | {lab} | " + " | ".join(cell(ms[k]) for k, _ in cols) + " |")
    lines.append("")
    lines.append("Each cell: count/n (rate, Wilson 95% interval in %). Paired naive_rag versus improved_rag "
                 "(good outcome on all questions):")
    for m, pr in s.get("paired", {}).items():
        g = pr["good_outcome_all"]
        lines.append(f"- {LABEL[m]}: naive {g['naive_rag']}/{g['n']}, improved {g['improved_rag']}/{g['n']}; "
                     f"discordant {g['naive_only']} naive only, {g['improved_only']} improved only; exact McNemar "
                     f"P = {g['mcnemar_exact_p']}; difference {100 * g['diff_improved_minus_naive']:.1f} points "
                     f"(95% CI {100 * g['diff_ci95_newcombe'][0]:.1f} to {100 * g['diff_ci95_newcombe'][1]:.1f}).")
    (a.summary.parent / "results_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", a.out_dir / "rag_eval.png", "and .pdf, .csv;", len(rows), "points; font",
          matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=plt.rcParams["font.sans-serif"])).split("/")[-1])


if __name__ == "__main__":
    main()
