"""A small LangChain chat model over a local transformers model, with a response cache.

``LocalChatModel`` is a ``BaseChatModel``, so it works in LCEL chains (``prompt | llm | parser``).
It overrides ``batch`` so that ``chain.batch(inputs)`` sends all prompts to the GPU together instead
of one at a time. Greedy decoding, so the same prompt always gives the same reply.

    from rag_lab.chat_model import LocalChatModel
    llm = LocalChatModel.from_model("Qwen/Qwen2.5-1.5B-Instruct", cache_path="cache/qwen15.jsonl")
    chain = prompt | llm | StrOutputParser()

The off-the-shelf alternative is ``langchain_huggingface.ChatHuggingFace`` over a
``HuggingFacePipeline``; this wrapper exists to batch on the GPU and to cache every reply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from .backends import Cached, HFLocal, model_key


def to_pair(messages: list[BaseMessage]) -> tuple[str, str]:
    system = "\n\n".join(m.content for m in messages if isinstance(m, SystemMessage))
    user = "\n\n".join(m.content for m in messages if isinstance(m, HumanMessage))
    return system, user


class LocalChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    backend: Any
    max_tokens: int = 200

    @classmethod
    def from_model(cls, model: str, cache_path: str | Path | None = None, max_tokens: int = 200,
                   batch_size: int = 16, seed: int = 0, offline: bool = False) -> "LocalChatModel":
        """Load a transformers model (hub id or local path), optionally behind a response cache.
        offline=True replays from the cache only and never loads the model."""
        if offline:
            return cls(backend=Cached(None, Path(cache_path), offline=True, name=model_key(model)),
                       max_tokens=max_tokens)
        be = HFLocal(model, batch_size=batch_size, seed=seed)
        if cache_path:
            be = Cached(be, Path(cache_path))
        return cls(backend=be, max_tokens=max_tokens)

    @property
    def _llm_type(self) -> str:
        return "local-transformers"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self.backend.complete_batch([to_pair(messages)], self.max_tokens)[0]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        pairs = [to_pair(self._convert_input(x).to_messages()) for x in inputs]
        texts = self.backend.complete_batch(pairs, self.max_tokens) if pairs else []
        return [AIMessage(content=t) for t in texts]
