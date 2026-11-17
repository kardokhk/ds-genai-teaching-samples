"""Settings for the RAG lab: models, revisions, seeds and retrieval parameters.

Model locations come from environment variables so the same code runs on Colab (hub ids, downloaded
on first use) and on a GPU server with a local model cache (snapshot paths, offline).

    LAB_CHAT_MODEL    chat model for the notebook (hub id or local path); default Qwen2.5-1.5B-Instruct
    LAB_EMBED_MODEL   embedding model; default BAAI/bge-small-en-v1.5
    LAB_RERANK_MODEL  cross-encoder reranker; default cross-encoder/ms-marco-MiniLM-L-6-v2

When HF_HUB_OFFLINE=1 and no override is set, models are loaded from the local Hugging Face cache by
snapshot path at the pinned revision (a cache filled by revision has no refs/main, so loading by hub
id fails offline).
"""
from __future__ import annotations

import os
from pathlib import Path

SEED = 20261117

MODELS = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
EMBED = ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
RERANK = ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# naive pipeline
NAIVE_CHUNK, NAIVE_OVERLAP, NAIVE_K = 1000, 100, 4
# improved pipeline
IMPROVED_CHUNK, IMPROVED_OVERLAP = 800, 100
TABLE_ROWS_PER_CHUNK = 6
HYBRID_K = 10          # candidates from each of BM25 and FAISS
HYBRID_WEIGHTS = (0.5, 0.5)
RERANK_TOP_N = 5
MAX_NEW_TOKENS = 200


def snapshot_path(repo_id: str, revision: str) -> str | None:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub")
    p = Path(hub).expanduser() / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    return str(p) if p.is_dir() else None


def locate(repo_id: str, revision: str, env_var: str | None = None) -> str:
    """Local override, then a cached snapshot, then the hub id (downloads on first use)."""
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    return snapshot_path(repo_id, revision) or repo_id


def chat_model_path(short: str) -> str:
    repo, rev = MODELS[short]
    return locate(repo, rev, None)


def embed_path() -> str:
    return locate(*EMBED, "LAB_EMBED_MODEL")


def rerank_path() -> str:
    return locate(*RERANK, "LAB_RERANK_MODEL")
