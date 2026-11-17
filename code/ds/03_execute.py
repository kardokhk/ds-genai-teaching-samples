"""Execute the learner notebook in a clean folder and save the executed copy.

The notebook runs from a folder that holds data/raw/winequality-red.csv, as it
does in the published repository, so the Colab setup cell is skipped.
"""
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "ds" / "notebooks" / "ds_first_classifier_lab.ipynb"
OUT = ROOT / "code" / "ds" / "notebooks" / "ds_first_classifier_lab.executed.ipynb"
WORK = ROOT / "build" / "ds_exec"

if WORK.exists():
    shutil.rmtree(WORK)
(WORK / "data" / "raw").mkdir(parents=True)
shutil.copy(ROOT / "data" / "raw" / "winequality-red.csv", WORK / "data" / "raw" / "winequality-red.csv")
nb = nbformat.read(SRC, as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(WORK)}}).execute()
for c in nb.cells:  # drop execution timestamps so reruns compare cleanly
    c.metadata.pop("execution", None)
nbformat.write(nb, OUT)
print("executed", OUT.relative_to(ROOT), file=sys.stderr)
