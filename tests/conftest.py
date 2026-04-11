"""pytest configuration for the tests/ directory.

Makes the repo root and training/ importable without requiring installation.
"""
import sys
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"

for p in (str(REPO_ROOT), str(TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
