"""Session analysis: first classifier on the UCI red wine data.

Computes every number and figure used in the Data Science sample session
(slides, instructor guide and notebook answers) and writes them to
results/<run>/numbers.json and figures/ds/.

Question taught: "Can we flag the wines tasters rate as good (quality 7 or
higher) from lab measurements, and how do we know if the model is any good?"

Run from the job folder:
    python code/ds/01_analysis.py
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "winequality-red.csv"
RUN = ROOT / "results" / "2026-11-17_ds_lesson"
FIG = ROOT / "figures" / "ds"

# Okabe-Ito
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#8A939B", "#1F2328"


def wilson(k, n, z=1.959964):
    """Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    out = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(n),
        "accuracy": accuracy_score(y, pred),
        "accuracy_ci": wilson(tp + tn, n),
        "recall": recall_score(y, pred, zero_division=0),
        "recall_ci": wilson(tp, tp + fn),
        "precision": precision_score(y, pred, zero_division=0),
        "precision_ci": wilson(tp, tp + fp) if (tp + fp) else (float("nan"), float("nan")),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "flagged": int(tp + fp),
    }
    return out


def slide_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.linewidth": 1.0, "savefig.dpi": 200,
        "svg.hashsalt": "ds", "pdf.fonttype": 42,
    })


def save(fig, name, data=None):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    if data is not None:
        data.to_csv(FIG / f"{name}.csv", index=False)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW, sep=";")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["good"] = (df["quality"] >= 7).astype(int)
    feats = [c for c in df.columns if c not in ("quality", "good")]
    X, y = df[feats], df["good"]

    N = {"seed": SEED, "n_rows": len(df), "n_features": len(feats), "features": feats,
         "n_good": int(y.sum()), "pct_good": float(y.mean()),
         "quality_counts": {int(k): int(v) for k, v in df["quality"].value_counts().sort_index().items()},
         "n_duplicate_rows": int(df.duplicated().sum()), "n_missing": int(df.isna().sum().sum())}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    N.update({"n_train": len(ytr), "n_test": len(yte), "n_test_good": int(yte.sum()),
              "n_train_good": int(ytr.sum())})

    # 1. Majority baseline: always say "not good".
    base = np.zeros_like(yte)
    N["baseline"] = metrics(yte, base)

    # 2. Decision tree, depth 1..15: train accuracy vs 5-fold CV accuracy on the training set.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for d in range(1, 16):
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        cv_acc = cross_val_score(t, Xtr, ytr, cv=cv, scoring="accuracy")
        t.fit(Xtr, ytr)
        rows.append({"depth": d, "train_accuracy": accuracy_score(ytr, t.predict(Xtr)),
                     "cv_accuracy": cv_acc.mean(), "cv_accuracy_sd": cv_acc.std(ddof=1)})
    depth = pd.DataFrame(rows)
    best = depth.loc[depth["cv_accuracy"].idxmax()]
    N["tree_depths"] = rows
    N["tree_best_depth"] = int(best["depth"])
    N["tree_best_cv_accuracy"] = float(best["cv_accuracy"])
    deep = DecisionTreeClassifier(max_depth=None, random_state=SEED).fit(Xtr, ytr)
    N["tree_unlimited"] = {"depth": int(deep.get_depth()),
                           "train_accuracy": accuracy_score(ytr, deep.predict(Xtr)),
                           "test": metrics(yte, deep.predict(Xte))}
    tree = DecisionTreeClassifier(max_depth=int(best["depth"]), random_state=SEED).fit(Xtr, ytr)
    N["tree_best_test"] = metrics(yte, tree.predict(Xte))

    # 3. Logistic regression with scaling.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr_cv = cross_val_score(lr, Xtr, ytr, cv=cv, scoring="accuracy")
    lr.fit(Xtr, ytr)
    proba = lr.predict_proba(Xte)[:, 1]
    N["logreg_cv_accuracy"] = float(lr_cv.mean())
    N["logreg_t50"] = metrics(yte, (proba >= 0.5).astype(int))
    N["logreg_t30"] = metrics(yte, (proba >= 0.3).astype(int))
    N["logreg_t20"] = metrics(yte, (proba >= 0.2).astype(int))
    coefs = pd.Series(lr[-1].coef_[0], index=feats).sort_values()
    N["logreg_std_coefficients"] = {k: float(v) for k, v in coefs.items()}

    # Exercise solutions: compare models by 5-fold CV balanced accuracy on the training set,
    # then report each on the test set once.
    from sklearn.ensemble import RandomForestClassifier
    cands = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "logreg_balanced": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "tree_unlimited": DecisionTreeClassifier(random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
    }
    ex = {}
    for name, m in cands.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="balanced_accuracy")
        m.fit(Xtr, ytr)
        ex[name] = {"cv_balanced_accuracy": float(s.mean()), "cv_balanced_accuracy_sd": float(s.std(ddof=1)),
                    "test": metrics(yte, m.predict(Xte))}
    N["exercise_models"] = ex

    thr = np.round(np.arange(0.05, 0.751, 0.05), 2)
    tcurve = []
    for t in thr:
        m = metrics(yte, (proba >= t).astype(int))
        tcurve.append({"threshold": float(t), "precision": m["precision"], "recall": m["recall"],
                       "flagged": m["flagged"]})
    tcurve = pd.DataFrame(tcurve)

    # Descriptive: alcohol by class (median and 10th/90th percentiles).
    alc = {}
    for lab, g in df.groupby("good")["alcohol"]:
        alc[int(lab)] = {"median": float(g.median()), "p10": float(g.quantile(0.10)),
                         "p90": float(g.quantile(0.90)), "n": int(g.size)}
    N["alcohol_by_class"] = alc

    # ---------------- figures for slides
    slide_style()
    # f1 class balance
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    counts = [N["n_rows"] - N["n_good"], N["n_good"]]
    ax.barh(["Not good (3 to 6)", "Good (7 or 8)"], counts, color=[GREY, BLUE], height=0.6)
    for i, c in enumerate(counts):
        ax.text(c + 15, i, f"{c:,} ({c / N['n_rows']:.0%})", va="center", fontsize=15)
    ax.set_xlim(0, max(counts) * 1.32)
    ax.set_xlabel("Number of wines")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "f1_class_balance", pd.DataFrame({"class": ["not_good", "good"], "n": counts}))

    # f2 alcohol by class
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rng = np.random.default_rng(SEED)
    for i, (lab, col, name) in enumerate([(0, GREY, "Not good"), (1, BLUE, "Good")]):
        v = df.loc[df.good == lab, "alcohol"].to_numpy()
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, v.size), s=7, alpha=0.35, color=col, lw=0)
        q = np.quantile(v, [0.1, 0.5, 0.9])
        ax.plot([q[0], q[2]], [i, i], color=INK, lw=2.2)
        ax.plot(q[1], i, "o", color="white", mec=INK, mew=2.2, ms=10)
    ax.set_yticks([0, 1], ["Not good", "Good"])
    ax.set_xlabel("Alcohol (% vol)")
    ax.set_ylim(-0.6, 1.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "f2_alcohol_by_class", df[["alcohol", "good"]])

    # f3 overfitting
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(depth.depth, depth.train_accuracy, "-o", color=ORANGE, ms=5, lw=2, label="Training data")
    ax.plot(depth.depth, depth.cv_accuracy, "-o", color=BLUE, ms=5, lw=2, label="Held-out folds (5-fold CV)")
    ax.axhline(1 - N["pct_good"], color=GREY, ls="--", lw=1.5)
    ax.text(15.4, 1 - N["pct_good"] + 0.006, "Always say \"not good\"", ha="right", va="bottom", fontsize=13, color=GREY)
    ax.set_xlabel("Tree depth (how many questions the tree may ask)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.8, 1.01)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.64))
    save(fig, "f3_tree_depth", depth)

    # f4 confusion matrices at two thresholds
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for ax, key, title in [(axes[0], "logreg_t50", "Threshold 0.5"), (axes[1], "logreg_t30", "Threshold 0.3")]:
        m = N[key]
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        shade = np.array([[0.10, 0.45], [0.45, 0.25]])
        ax.imshow(shade, cmap="Greys", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, f"{v}", ha="center", va="center", fontsize=20, color=INK, weight="bold")
        ax.set_xticks([0, 1], ["Predicted\nnot good", "Predicted\ngood"])
        ax.set_yticks([0, 1], ["Actually\nnot good", "Actually\ngood"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{title}: recall {m['recall']:.0%}, precision {m['precision']:.0%}", fontsize=14)
    fig.tight_layout(w_pad=3)
    save(fig, "f4_confusion", None)

    # f5 threshold curve
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(tcurve.threshold, tcurve.recall, "-o", ms=4, lw=2, color=BLUE, label="Recall (good wines found)")
    ax.plot(tcurve.threshold, tcurve.precision, "-o", ms=4, lw=2, color=ORANGE, label="Precision (flags that are right)")
    ax.axvline(0.5, color=GREY, ls=":", lw=1.4)
    ax.axvline(0.3, color=GREY, ls=":", lw=1.4)
    ax.set_xlabel("Decision threshold on predicted probability")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 0.78)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=13)
    save(fig, "f5_threshold", tcurve)

    # f6 coefficients
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    nice = {k: k.replace("_", " ") for k in coefs.index}
    cols = [BLUE if v > 0 else ORANGE for v in coefs.values]
    ax.barh([nice[k] for k in coefs.index], coefs.values, color=cols, height=0.65)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Standardised coefficient (log-odds per 1 SD)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "f6_coefficients", coefs.rename("coef").reset_index().rename(columns={"index": "feature"}))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 6)
        if isinstance(o, np.integer):
            return int(o)
        return o

    N["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit_learn": __import__("sklearn").__version__, "matplotlib": matplotlib.__version__}
    (RUN / "numbers.json").write_text(json.dumps(clean(N), indent=1) + "\n")
    print(json.dumps({k: clean(N[k]) for k in ["n_rows", "n_good", "pct_good", "n_test", "n_test_good",
                                               "tree_best_depth", "tree_best_cv_accuracy", "logreg_cv_accuracy"]}))
    for k in ["baseline", "tree_best_test", "logreg_t50", "logreg_t30", "logreg_t20"]:
        m = N[k]
        print(k, f"acc {m['accuracy']:.3f} rec {m['recall']:.3f} prec {m['precision']:.3f} bal {m['balanced_accuracy']:.3f}",
              m["tp"], m["fp"], m["fn"], m["tn"])
    print("unlimited", N["tree_unlimited"]["depth"], round(N["tree_unlimited"]["train_accuracy"], 3),
          round(N["tree_unlimited"]["test"]["accuracy"], 3))
    print(coefs.round(2).to_dict())
    print(alc)


if __name__ == "__main__":
    main()
