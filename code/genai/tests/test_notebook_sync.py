"""The grounded prompt taught in the notebook is the one that was evaluated."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rag_lab.pipelines import GROUNDED_SYSTEM  # noqa: E402


def test_notebook_prompt_matches_pipeline():
    nb = json.loads((HERE / "notebooks" / "genai_rag_lab.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert GROUNDED_SYSTEM in src


def test_single_colab_setup_cell():
    nb = json.loads((HERE / "notebooks" / "genai_rag_lab.ipynb").read_text())
    tagged = [c for c in nb["cells"] if "colab-setup" in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1 and "git clone" in "".join(tagged[0]["source"])
    others = "\n".join("".join(c["source"]) for c in nb["cells"] if c not in tagged)
    assert "git clone" not in others and "github.com" not in others
