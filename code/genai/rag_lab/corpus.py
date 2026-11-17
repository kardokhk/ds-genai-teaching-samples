"""Load the markdown knowledge base as LangChain Documents, keeping the front matter as metadata."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    _, head, body = text.split("---", 2)
    meta = {}
    for line in head.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, body.lstrip("\n")


def load_documents(corpus_dir: str | Path) -> list[Document]:
    docs = []
    for p in sorted(Path(corpus_dir, "docs").glob("*.md")):
        meta, body = parse_front_matter(p.read_text())
        meta["source"] = p.name
        docs.append(Document(page_content=body, metadata=meta))
    if not docs:
        raise FileNotFoundError(f"no markdown documents in {Path(corpus_dir, 'docs')}")
    return docs


def load_questions(corpus_dir: str | Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(corpus_dir, "questions.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if split in (None, "all") or r["split"] == split]
