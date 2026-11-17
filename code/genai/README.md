# RAG teaching lab: a company knowledge base with LangChain

A code-along lab for a three-hour live session on retrieval-augmented generation (RAG). Learners build a question-answering service over the internal documents of a fictional company, Larkspur Freight Ltd, and then measure it.

The lab compares three pipelines, each written in LangChain:

| Pipeline | What it does |
|---|---|
| closed_book | the chat model alone, no documents |
| naive_rag | 1,000-character chunks, dense search (bge-small and FAISS), top 4, plain "answer using the context" prompt |
| improved_rag | heading-aware chunks with metadata; superseded documents filtered out; hybrid search (BM25 and FAISS through `EnsembleRetriever`); cross-encoder rerank (`ContextualCompressionRetriever` and `CrossEncoderReranker`); grounded prompt with numbered sources, a citation after each sentence, `NOT_FOUND` when the sources do not answer, and document text treated as data |

All chains use LCEL (`prompt | llm | parser`). The chat model is a small LangChain `BaseChatModel` over transformers (`rag_lab/chat_model.py`) that batches on the GPU, decodes greedily and caches every reply.

## Results on the held-out test questions

One frozen run on 45 held-out questions (37 answerable, 8 unanswerable), run id `frozen-b0afb587e55e`. Numbers are copied from `results/2026-11-17_rag_eval/summary.json`; the figure is `figures/genai/rag_eval.png`.

| Model | Pipeline | Correct (answerable) | Refuses (unanswerable) | Confident wrong (all) | Cites right document | Old policy value | Follows injection |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Closed book | 0/37 (0%, 0 to 9) | 6/8 (75%, 41 to 93) | 19/45 (42%, 29 to 57) | 0/37 (0%, 0 to 9) | 1/8 (12%, 2 to 47) | 0/3 (0%, 0 to 56) |
| Qwen2.5-1.5B-Instruct | Naive RAG | 28/37 (76%, 60 to 87) | 5/8 (62%, 31 to 86) | 12/45 (27%, 16 to 41) | 0/37 (0%, 0 to 9) | 5/8 (62%, 31 to 86) | 1/3 (33%, 6 to 79) |
| Qwen2.5-1.5B-Instruct | Improved RAG | 34/37 (92%, 79 to 97) | 8/8 (100%, 68 to 100) | 2/45 (4%, 1 to 15) | 11/37 (30%, 17 to 46) | 0/8 (0%, 0 to 32) | 0/3 (0%, 0 to 56) |
| Qwen2.5-7B-Instruct | Closed book | 1/37 (3%, 0 to 14) | 3/8 (38%, 14 to 69) | 34/45 (76%, 61 to 86) | 0/37 (0%, 0 to 9) | 1/8 (12%, 2 to 47) | 0/3 (0%, 0 to 56) |
| Qwen2.5-7B-Instruct | Naive RAG | 29/37 (78%, 63 to 89) | 7/8 (88%, 53 to 98) | 7/45 (16%, 8 to 29) | 0/37 (0%, 0 to 9) | 5/8 (62%, 31 to 86) | 1/3 (33%, 6 to 79) |
| Qwen2.5-7B-Instruct | Improved RAG | 35/37 (95%, 82 to 98) | 7/8 (88%, 53 to 98) | 1/45 (2%, 0 to 12) | 37/37 (100%, 91 to 100) | 0/8 (0%, 0 to 32) | 0/3 (0%, 0 to 56) |

Each cell: count/n (rate, Wilson 95% interval in %). Paired naive_rag versus improved_rag (good outcome on all questions):
- Qwen2.5-1.5B-Instruct: naive 33/45, improved 42/45; discordant 1 naive only, 10 improved only; exact McNemar P = 0.0117; difference 20.0 points (95% CI 5.7 to 34.4).
- Qwen2.5-7B-Instruct: naive 36/45, improved 42/45; discordant 1 naive only, 7 improved only; exact McNemar P = 0.0703; difference 13.3 points (95% CI 0.4 to 27.0).

What the run shows:
- Without documents, both models invent company facts: confident wrong answers on 19/45 questions (1.5B) and 34/45 (7B).
- Naive RAG gave the superseded policy value on 5/8 old-versus-new questions with either model; the metadata filter brought this to 0/8.
- Improved RAG cut confident wrong answers to 2/45 (1.5B) and 1/45 (7B).
- The 7B model cited the right document on 37/37 answerable questions; the 1.5B model often left out the citation (11/37).
- Weak points: the two-document trunk-run question (customer price plus supplier rate) failed with both models in improved RAG, and on the dev questions the 1.5B model repeated a planted code in both injection tests even with the grounded prompt.
- Three correct answers to one question (Q062) were under-credited by the frozen grader; they are listed in `summary.json` and the numbers above were not changed.

## What is in this folder

| Path | Purpose |
|---|---|
| `01_make_corpus.py` | writes the 12 markdown documents, `facts.json` and `manifest.json` (seeded, deterministic) |
| `02_make_questions.py` | writes `questions.jsonl`, with gold answers read from `facts.json` and the dev/test split |
| `03_evaluate.py` | runs and grades the pipelines (`run`), then summarises with intervals and a paired test (`summarise`) |
| `04_figure.py` | figure of the test run, with the plotted values as CSV |
| `05_grader_check.py` | compares the grader with hand scores |
| `rag_lab/` | config, model backends and response cache, LangChain chat model, corpus loader, pipelines, grader, statistics |
| `notebooks/genai_rag_lab.ipynb` | the learner notebook; `genai_rag_lab.executed.ipynb` is the same notebook run top to bottom on a GPU |
| `notebooks/make_notebook.py`, `execute_notebook.py` | build the notebook from Python, and execute it |
| `slurm/rag_eval.sbatch` | batch job for a GPU cluster (set your own account) |
| `tests/` | tests for the corpus, the grader, the statistics and the notebook |

## Run on Colab

1. Open `notebooks/genai_rag_lab.ipynb` in Colab and choose a GPU runtime (a T4 is enough for Qwen2.5-1.5B-Instruct).
2. Run the first two cells: they download the lab code and install the LangChain packages.
3. Run the rest top to bottom. The knowledge base is generated on the spot if it is not present, and is identical every time.

To use another model, set `LAB_CHAT_MODEL` to a hub id or a local folder before loading it.

## Run on a GPU server

```bash
pip install -r requirements.txt
python 01_make_corpus.py && python 02_make_questions.py
pytest -q tests
# dry run without a GPU: a scripted stand-in model, 3 questions per pipeline
python 03_evaluate.py run --split dev --model scripted --backend scripted --limit 3 --run-dir /tmp/dry
# full run with one model on one GPU
python 03_evaluate.py run --split test --model qwen2.5-7b --backend hf --run-dir runs/test
python 03_evaluate.py summarise --run-dir runs/test --out runs/test/summary.json
```

On a Slurm cluster, `slurm/rag_eval.sbatch` runs both models on two GPUs of one node (and optionally executes the notebook on a third). Replace `YOUR_ACCOUNT` and export `PY` (your Python) and `HF_HOME` (your model cache) before `sbatch`. With `HF_HUB_OFFLINE=1`, models are loaded from the local cache by snapshot path at the pinned revisions below.

## Replay the evaluation on a CPU, without a GPU

Every model reply is stored in `results/2026-11-17_rag_eval/cache/<model>.jsonl`, keyed on the model, the prompt and the token limit. Retrieval always runs on the CPU, so a replay builds the same prompts and finds each reply in the cache. A missing reply is an error, never a silent difference.

```bash
python 03_evaluate.py run --split test --model qwen2.5-7b --backend replay --run-dir runs/replay
python 03_evaluate.py run --split test --model qwen2.5-1.5b --backend replay --run-dir runs/replay
python 03_evaluate.py summarise --run-dir runs/replay --out runs/replay/summary.json
```

## Models, seeds and settings

| Role | Model | Revision | Licence |
|---|---|---|---|
| Chat (small) | Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | Apache-2.0 |
| Chat (large) | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | Apache-2.0 |
| Embeddings | BAAI/bge-small-en-v1.5 | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a | MIT |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a | Apache-2.0 |

Seed 20261117 for the corpus, numpy, torch and transformers; greedy decoding; at most 200 new tokens; bf16 on GPU. Retrieval settings are in `rag_lab/config.py`.

## How the evaluation works

- **Questions.** 66 questions with gold answers written from the generator's own values: single facts, table lookups (SKU, city, ticket priority), old-versus-new policy, two-document questions, injection bait, and 12 questions the documents cannot answer.
- **Split by document group.** The IT security, onboarding and operations-meeting documents form the dev split (21 questions); prompts and settings were tuned on dev only. The leave, remote-work, travel, price list, SLA, supplier and support-meeting documents form the test split (45 questions), run once after the code was frozen.
- **Grader.** Plain Python, no model (`rag_lab/grader.py`). A correct answer contains every gold value and does not obey a planted instruction; a correct refusal says the information is not available; a confident wrong answer neither refuses nor is correct. Before the test run, the grader was compared with hand scores on 30 dev outputs and fixed where it was wrong. After the frozen run, a fresh sample of 30 test outputs was hand scored with the grader unchanged: 29/30 agreed.
- **Intervals.** Wilson 95% intervals for each rate. naive_rag and improved_rag are compared on the same questions with an exact McNemar test and a Newcombe 95% interval for the paired difference.

## Limitations

- The question set is small (45 test questions), so intervals are wide; a difference of one or two questions is not meaningful.
- The corpus is fictional and short (12 documents). Real knowledge bases have longer documents, scanned PDFs and conflicting owners, and retrieval is harder.
- The grader checks that the required values are present, not that every other sentence is true, and a citation counts as right when it points to a gold document, not when it supports each sentence.
- The injection test uses two planted lines of one style. Passing it does not show resistance to other attacks; actions such as refunds belong in code with human approval, not in a prompt.
- This is a teaching prototype, not a production system.
