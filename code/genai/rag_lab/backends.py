"""Batch text-generation backends with a response cache for GPU-free replay.

``complete_batch(prompts, max_tokens) -> list[str]`` with prompts as (system, user) pairs.

* ``HFLocal``   a local open-weight model run with transformers: batched, left-padded, greedy.
* ``Scripted``  deterministic replies from a Python function, for tests and dry runs.
* ``Cached``    wraps either; one append-only JSONL file keyed on the model name, the prompt and
                the token limit. ``offline=True`` raises on any miss, so a replay never silently
                differs from the recorded run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

Prompt = tuple[str, str]  # (system, user)


def model_key(model: str) -> str:
    """Same cache key for a local snapshot path and a hub id: models--Org--Name."""
    parts = Path(model).parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") - 1]
    return "models--" + model.replace("/", "--")


class HFLocal:
    def __init__(self, model: str, batch_size: int = 16, dtype: str = "bfloat16", device_map: str = "auto",
                 seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(seed)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if not torch.cuda.is_available():
            dtype, device_map = "float32", "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map=device_map)
        self.model.eval()
        self.bs = batch_size
        self.name = model_key(model)

    def complete_batch(self, prompts: Sequence[Prompt], max_tokens: int = 200) -> list[str]:
        out: list[str] = [""] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i][0]) + len(prompts[i][1]))
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            texts = []
            for i in idx:
                msgs = ([{"role": "system", "content": prompts[i][0]}] if prompts[i][0] else []) + \
                       [{"role": "user", "content": prompts[i][1]}]
                texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, temperature=None,
                                          top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id)
            for j, i in enumerate(idx):
                out[i] = self.tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return out


class Scripted:
    def __init__(self, fn: Callable[[str, str], str], name: str = "scripted"):
        self.fn, self.calls, self.name = fn, [], name
        self.batches: list[int] = []

    def complete_batch(self, prompts, max_tokens=200):
        self.calls.extend(prompts)
        self.batches.append(len(prompts))
        return [self.fn(s, u) for s, u in prompts]


class Cached:
    """Response cache keyed by model, prompt and token limit."""

    def __init__(self, backend, path: Path, offline: bool = False, name: str | None = None):
        self.backend, self.path, self.offline = backend, Path(path), offline
        if name is None and backend is None:
            raise ValueError("an offline cache needs the model name it was recorded with (model_key)")
        self.name = name or getattr(backend, "name", "unknown")
        self.mem: dict[str, str] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.mem[rec["k"]] = rec["v"]

    def key(self, p: Prompt, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([self.name, p[0], p[1], max_tokens]).encode()).hexdigest()

    def complete_batch(self, prompts, max_tokens=200):
        keys = [self.key(p, max_tokens) for p in prompts]
        todo, seen = [], set()
        for i, k in enumerate(keys):  # one model call per distinct prompt not yet cached
            if k not in self.mem and k not in seen:
                todo.append(i)
                seen.add(k)
        self.hits += len(prompts) - len(todo)
        self.misses += len(todo)
        if todo:
            if self.offline or self.backend is None:
                raise RuntimeError(f"{len(todo)} prompts not in cache {self.path} and no model loaded")
            res = self.backend.complete_batch([prompts[i] for i in todo], max_tokens)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                for i, v in zip(todo, res):
                    self.mem[keys[i]] = v
                    f.write(json.dumps({"k": keys[i], "v": v}) + "\n")
        return [self.mem[k] for k in keys]
