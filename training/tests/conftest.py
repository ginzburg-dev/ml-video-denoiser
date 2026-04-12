"""pytest configuration — adds training/ to sys.path for all tests."""

import sys
from pathlib import Path

# Allow imports like `from models import UNetResidual` from within tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
