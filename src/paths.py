"""Folder layout shared by every script. All paths are absolute, so scripts can be run from anywhere.

Everything the scripts write goes under RESULTS (results/ by default, git-ignored), the
figures included, because all of it is derived from the licensed data. Set the environment
variable REPLICATION_RESULTS to write results somewhere other than results/.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA = DATA_DIR / "data.xlsx"          # licensed input, not in the repository (see data/README.md)
RESULTS = Path(os.environ.get("REPLICATION_RESULTS", ROOT / "results")).resolve()   # git-ignored

FIGURES = RESULTS / "figures"              # the thesis figures (PNG)
CACHE = RESULTS / "cache"                  # ridge-path caches
ORIGINAL_GRID = RESULTS / "original_grid"  # tuning grid of the original specification (optional)
ROBUSTNESS = RESULTS / "robustness"        # original-specification runs and tab:original
GRIDS = RESULTS / "grids"                  # final-specification tuning grids and the selection
FINAL = RESULTS / "final"                  # final runs, tests and tables.json
FINAL_SIBLINGS = RESULTS / "final_siblings"  # sibling-category runs
FIGURE_DATA = RESULTS / "figure_data"      # numbers behind the figures
LOGS = RESULTS / "logs"                    # one log per stage of run_all.py


def ensure(path):
    """Create a folder if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
